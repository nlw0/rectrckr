#include "vectors.h"
#include "camera_models.h"
#include "target.h"

Vector2D p(const Vector3D& s,
                         const Vector3D& t,
                         const Quaternion& psi,
                         const CameraModel& cm) {
  return project(q(s, t, psi), cm);
}

Vector3D q(const Vector3D& s, const Vector3D& t, const Quaternion& psi) {
  Vector3D output;
  Vector3D rx = r(psi, 0);
  Vector3D ry = r(psi, 1);
  Vector3D rz = r(psi, 2);
  Vector3D d = s - t;

  return Vector3D(d * rx, d * ry, d * rz);
}

Vector3D dq_ds(const Quaternion& psi, const int& k) {
  Vector3D output;
  Vector3D rx = r(psi, 0);
  Vector3D ry = r(psi, 1);
  Vector3D rz = r(psi, 2);

  if (k==0){
    return Vector3D(rx.x, ry.x, rz.x);
  } else if (k==1) {
    return Vector3D(rx.y, ry.y, rz.y);
  } else if (k==2) {
    return Vector3D(rx.z, ry.z, rz.z);
  } else {
    return Vector3D(0, 0, 0);
  }
}

Vector3D dq_dt(const Quaternion& psi, const int& k) {
  return -1.0 * dq_ds(psi, k);
}

Vector3D dq_dpsi(const Vector3D& s, const Vector3D& t, const Quaternion& psi, const int& k) {
  Vector3D output;
  Vector3D rx_ = dr_dpsi(psi, 0, k);
  Vector3D ry_ = dr_dpsi(psi, 1, k);
  Vector3D rz_ = dr_dpsi(psi, 2, k);
  Vector3D d = s - t;

  return Vector3D(d * rx_, d * ry_, d * rz_);
}

/* Reference frame vector at the specified direction. */
Vector3D r(Quaternion psi, int direction) {
  return Vector3D( \
    (direction==0) ? (psi.a*psi.a+psi.b*psi.b-psi.c*psi.c-psi.d*psi.d) :
    (direction==1) ? (2*psi.b*psi.c-2*psi.a*psi.d) :
    (direction==2) ? (2*psi.b*psi.d+2*psi.a*psi.c) : 0,
    (direction==0) ? (2*psi.b*psi.c+2*psi.a*psi.d) :
    (direction==1) ? (psi.a*psi.a-psi.b*psi.b+psi.c*psi.c-psi.d*psi.d) :
    (direction==2) ? (2*psi.c*psi.d-2*psi.a*psi.b) : 0,
    (direction==0) ? (2*psi.b*psi.d-2*psi.a*psi.c) :
    (direction==1) ? (2*psi.c*psi.d+2*psi.a*psi.b) :
    (direction==2) ? (psi.a*psi.a-psi.b*psi.b-psi.c*psi.c+psi.d*psi.d) : 0
  );
}

/* Derivative of each vector component relative to the quaternion
   component k. */
Vector3D dr_dpsi(Quaternion psi, int direction, int k) {
  Vector3D output;
  output.x =
    (direction==0) ?
    ((k==0) ?  psi.a :
     (k==1) ?  psi.b :
     (k==2) ? -psi.c :
     (k==3) ? -psi.d : 0) :
    (direction==1) ?
    ((k==0) ? -psi.d :
     (k==1) ?  psi.c :
     (k==2) ?  psi.b :
     (k==3) ? -psi.a : 0) :
    (direction==2) ?
    ((k==0) ?  psi.c :
     (k==1) ?  psi.d :
     (k==2) ?  psi.a :
     (k==3) ?  psi.b : 0) : 0;

  output.y = 
    (direction==0) ?
    ((k==0) ?  psi.d :
     (k==1) ?  psi.c :
     (k==2) ?  psi.b :
     (k==3) ?  psi.a : 0) :
    (direction==1) ?
    ((k==0) ?  psi.a :
     (k==1) ? -psi.b :
     (k==2) ?  psi.c :
     (k==3) ? -psi.d : 0) :
    (direction==2) ?
    ((k==0) ? -psi.b :
     (k==1) ? -psi.a :
     (k==2) ?  psi.d :
     (k==3) ?  psi.c : 0) : 0;

  output.z =
    (direction==0) ?
    ((k==0) ? -psi.c :
     (k==1) ?  psi.d :
     (k==2) ? -psi.a :
     (k==3) ?  psi.b : 0) :
    (direction==1) ?
    ((k==0) ?  psi.b :
     (k==1) ?  psi.a :
     (k==2) ?  psi.d :
     (k==3) ?  psi.c : 0) :
    (direction==2) ?
    ((k==0) ?  psi.a :
     (k==1) ? -psi.b :
     (k==2) ? -psi.c :
     (k==3) ?  psi.d : 0) : 0;

  return output;
}
