// Copyright 2011 Nicolau Leal Werneck, Anna Helena Reali Costa and
// Universidade de São Paulo
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include<stdio.h>

// file corisco_aux_external.c
//
// Contains function SSE_rsqrt to calculate reciprocal of square root
// using SSE helper instruction plus a single Newton-Raphson iteration.
//
// Totally ripped off from example posted by dude at stackoverflow
// To use with Cython, compile with gcc -fPIC -c rsqrt.c -o rsqrt.o
// stolen and ressocialized by nwerneck@gmail.com - 16/09/2010

#include<emmintrin.h>
#include<xmmintrin.h>

#include<math.h>
#include<tgmath.h>


/** These macros use inline asm to apply the rsqrt isntruction to calculate the reverse square root ofa  value in either a variable that will be the output, or an expression result, etc (the "2" version).*/
#define NIC_SSE_RSQRT(A)   asm ("rsqrtps %[a], %[a]" : [a] "=x"((A)) : "x"((A)) );
#define NIC_SSE_RSQRT2(B,A)   asm ("rsqrtps %0, %0" : "=x"((B)) : "x"((A)) );

#define VEC_ASSIGN(var, val) (var).f[0]=(val);(var).f[1]=(val);(var).f[2]=(val);(var).f[3]=(val);

typedef float v4sf __attribute__ ((  vector_size (16) )); // vector of four single floats
  
union f4vector 
{
  v4sf v;
  float f[4];
};


/* Uses rsqrt instruction intrinsic to calculate reciprocal of square root */
inline void SSE_rsqrt( float *pOut, float *pIn )
{
    _mm_store_ss( pOut, _mm_rsqrt_ss( _mm_load_ss( pIn ) ) );
}

/* Gets rsqrt estimate then perform as single step of the
   Newton-Raphson iteration. */
inline void SSE_rsqrt_NR( float *pOut, float *pIn )
{
    _mm_store_ss( pOut, _mm_rsqrt_ss( _mm_load_ss( pIn ) ) );
    if (*pOut < 1e10) {
      *pOut *= ((3.0f - *pOut * *pOut * *pIn) * 0.5f);
    } else {
      *pOut = 1e10;
    }
}


/** SSE Vector operations */
inline void myvec_sumacc(float*a, float*b) {
  (*(union f4vector*)a).v +=   (*(union f4vector*)b).v;
}
inline void myvec_mulacc(float*a, float*b) {
  (*(union f4vector*)a).v *=   (*(union f4vector*)b).v;
}
inline void myvec_pos_lim(float*a) {
  static float z[4] = {0.0f,0.0f,0.0f,0.0f};
  (*(__m128*)a) = _mm_max_ps((*(__m128*)a) , (*(__m128*)z) );
}
inline void myvec_abs(float*a) {
  (*(__m128*)a) = _mm_max_ps((*(__m128*)a) , -(*(__m128*)a) );
}
inline void myvec_copy(float*a, float*b) {
  (*(__m128*)a) = (*(__m128*)b);
}
inline void myvec_rsqrt(float*a) {
  (*(__m128*)a) = _mm_rsqrt_ps((*(__m128*)a));
}
inline void myvec_rcp(float*a) {
  (*(__m128*)a) = _mm_rcp_ps((*(__m128*)a));
}

/** Some floating-pount ioperations */
inline void myabs(float* x){
  *x = (*x>0)?*x:-*x;
}
inline void mypos_lim(float* x){
  *x = (*x>0)?*x:0;
}
/** */





/** Reads clock time (it's a x86_64 thing) */
static __inline__ unsigned long long rdtsc(void)
{
  unsigned hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}





float the_big_MAP_loop(
		       float* vp,
		       float middlex, float middley, float fd,
		       float* mask_mem, float*gradx_mem, float*grady_mem,
		       unsigned int ymax, unsigned int xmax, float param3
		       )
{
  /** This is where we accumulate the sum to be returned at the end */
  float the_sum = 0;
  
  
  /** Componentes of directions of each of the 3 VPs. */
  union f4vector rx, ry, rz;
  /** The "characteristic vector" of each of the 3 VPs. */
  union f4vector rrx, rry;

  /** Create the vectorial versions of fd, middlex and middley. */
  union f4vector vec_fd, vec_middlex, vec_middley;
  VEC_ASSIGN(vec_fd, fd);
  VEC_ASSIGN(vec_middlex, middlex);
  VEC_ASSIGN(vec_middley, middley);

  /** Parameters for the angle function. */
  float sa = 2.0/param3;
  float sb = 2.0/(param3*param3);


  /** Print input VP directions */
  /* int j,k; */
  /* for (j=0; j<3; j++) { */
  /*   for (k=0; k<3; k++) */
  /* 	  printf("c %+.7e\t", vp[j*3+k]); */
  /*   printf("\n"); */
  /* } */
  /* printf("r\n"); */

  /** Move each vp direction vector into the x y and z component arrays. */
  rx.f[0] = vp[0];
  rx.f[1] = vp[3];
  rx.f[2] = vp[6];
  ry.f[0] = vp[1];
  ry.f[1] = vp[4];
  ry.f[2] = vp[7];
  rz.f[0] = vp[2];
  rz.f[1] = vp[5];
  rz.f[2] = vp[8];


  /** Calculate rrx and rry, used in the loop to calculate the
   *  expected VP directions. */
  rrx.v = vec_fd.v * rx.v/rz.v+vec_middlex.v;
  rry.v = vec_fd.v * ry.v/rz.v+vec_middley.v;


  /** Loop variables, line and column of image. These are "scalars"
      instead of vectors, therefore the "s". */
  unsigned int sy,sx;

  /** Vector versions of the variables... */
  union f4vector x, y;

  /** Auxiliary variable to hold an intermediate result. Just a silly premature optimization... :( */
  union f4vector pyout;

  /** Values related to the angle likelyhood function, plus the
      intermediate storage for the angle likelyhood to be multiplied
      by the mask. */
  union f4vector a, b, valz;
  VEC_ASSIGN(a, sa);
  VEC_ASSIGN(b, sb);
  /** It's just healthy to set the last values to 0 since they are not used. */
  a.f[3]=0;
  b.f[3]=0;
  valz.f[3]=0;
  


  /** VP directions at this point. */
  union f4vector px, py;

  /** When looping over the directions, store the value in the vector
      for multiplyong over all channels. */
  union f4vector px_aux, py_aux;

  /** Vectors to read the gradient values to. */
  union f4vector gradx, grady, mask;
  gradx.f[3]=0;
  grady.f[3]=0;
  mask.f[3]=0;

  /** Stores inverse square root of magnitude to normalize VP
      direction vector. */
  union f4vector rmag;
 
      
  /** The loop over each pixel.  Should be eventually replaced by a
      loop over a grid of pixels.  */
  for (sy=0; sy<ymax; sy++){
    VEC_ASSIGN(y, sy);
    /** intermediate result... Perhaps not much necessary
	optimization, but anyway. */
    pyout.v = rry.v - y.v;

    for (sx=0; sx<xmax; sx++){
      VEC_ASSIGN(x, sx);

      px.v = rrx.v - x.v;

      NIC_SSE_RSQRT2(rmag.v, px.v*px.v+pyout.v*pyout.v);

      /** Finally, px and py are unitary vectors in the three
	  vanishing point directions. */
      px.v *= rmag.v;
      py.v  = rmag.v*pyout.v;
      
      
      /** Load gradient values. */
      gradx.f[0] = gradx_mem[sy*xmax*3+sx*3  ];
      grady.f[0] = grady_mem[sy*xmax*3+sx*3  ];
      gradx.f[1] = gradx_mem[sy*xmax*3+sx*3+1];
      grady.f[1] = grady_mem[sy*xmax*3+sx*3+1];
      gradx.f[2] = gradx_mem[sy*xmax*3+sx*3+2];
      grady.f[2] = grady_mem[sy*xmax*3+sx*3+2];

      /** Load mask values into mask_abs, then multiply by angle
	  likelyhoods and sum. */
      mask.f[0] = mask_mem[sy*xmax*3+sx*3  ];
      mask.f[1] = mask_mem[sy*xmax*3+sx*3+1];
      mask.f[2] = mask_mem[sy*xmax*3+sx*3+2];

      /** Now loop over the VPs, and get statistics from three
	  channels simultaneously. */
      unsigned int dir;
      for (dir=0; dir<3; dir++){
	VEC_ASSIGN(px_aux, px.f[dir]);
	VEC_ASSIGN(py_aux, py.f[dir]);
	
	/** Each value in the valz vector refers to an image channel. */
	valz.v = px_aux.v * gradx.v + py_aux.v  * grady.v;
	myvec_abs(valz.f);
	valz.v = a.v+b.v*valz.v;
	myvec_pos_lim(valz.f);
		

	valz.v=mask.v * valz.v;
	
	the_sum+=valz.f[0]+valz.f[1]+valz.f[2];	

      }

      
      
      
      
      
      
    }
  }

  
    
  return the_sum;
}
 


