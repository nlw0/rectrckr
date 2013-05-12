#ifndef VECTORS_H
#define VECTORS_H

#include <math.h>

inline double rsqrt(double x) { return 1.0 / sqrtf(x); };

struct Vector2D {

  double x;
  double y;

  Vector2D() {}

  Vector2D(double x_, double y_)
    : x(x_), y(y_) {}

  Vector2D(double *p)
    : x(p[0]), y(p[1]) {}

  inline Vector2D& operator= (const Vector2D& b) {
    x = b.x;
    y = b.y;
    return (*this);
  }

  inline double operator[] (long k) {
    return (*this)[k];
  }
  inline const double operator[] (long k) const {
    return (*this)[k];
  }
  inline Vector2D operator+ (const Vector2D& b) const {
    return Vector2D(x + b.x, y + b.y);
  }
  inline Vector2D operator- () const {
    return Vector2D(-x, -y);
  }
  inline Vector2D operator- (const Vector2D& b) const {
    return Vector2D(x - b.x, y - b.y);
  }
  /* Dot/scalar/inner product */
  inline double operator* (const Vector2D& b) const {
    return x * b.x + y * b.y;
  }
  inline Vector2D operator* (const double& alpha) const {
    return Vector2D(alpha * x, alpha * y);
  }
  inline Vector2D operator/ (const double& alpha) const {
    return (*this) * (1.0 / alpha);
  }
  inline double norm () const {
    return sqrtf(x * x + y * y);
  }
  /* Cross/vector product */
  inline double operator% (const Vector2D& b) const {
    return x * b.y - y * b.x;
  }
  /* Hadamard/Schur/entrywise product */
  inline Vector2D operator& (const Vector2D& b) const {
    return Vector2D(x * b.x, y * b.y);
  }

};

inline Vector2D operator* (const double& alpha, const Vector2D& a) {
  return a * alpha;
}


struct Vector3D {
  
  double x;
  double y;
  double z;
  
  Vector3D() {}  

  Vector3D(double x_, double y_, double z_)
    : x(x_), y(y_), z(z_) {}

  Vector3D(double* p)
    : x(p[0]), y(p[1]), z(p[2]) {}

  inline double operator[] (long k) {
    return (*this)[k];
  }
  inline const double operator[] (long k) const {
    return (*this)[k];
  }
  inline Vector3D& operator= (const Vector3D& b) {
    x = b.x;
    y = b.y;
    z = b.z;
    return (*this);
  }
  inline Vector3D operator+ (const Vector3D& b) const {
    return Vector3D(x + b.x, y + b.y, z + b.z);
  }
  inline Vector3D operator- () const {
    return Vector3D(-x, -y, -z);
  }
  inline Vector3D operator- (const Vector3D& b) const {
    return Vector3D(x - b.x, y - b.y, z - b.z);
  }
  /* Dot/scalar/inner product */
  inline double operator* (const Vector3D& b) const {
    return x * b.x + y * b.y + z * b.z;
  }
  inline Vector3D operator* (const double& alpha) const {
    return Vector3D(alpha * x, alpha * y, alpha * z);
  }
  inline Vector3D operator/ (const double& alpha) const {
    return (*this) * (1.0 / alpha);
  }
  inline double norm () const {
    return sqrtf(x * x + y * y + z * z);
  }
  inline double inv_norm () const {
    return rsqrt(x * x + y * y + z * z);
  }
  /* Cross/vector product */
  inline Vector3D operator% (const Vector3D& b) const {
    return Vector3D(y * b.z - z * b.y,
                    z * b.x - x * b.z,
                    x * b.y - y * b.x);
  }
  /* Hadamard/Schur/entrywise product */
  inline Vector3D operator& (const Vector3D& b) const {
    return Vector3D(x * b.x, y * b.y, z * b.z);
  }

}; 

inline Vector3D operator* (const double& alpha, const Vector3D& a) {
  return a * alpha;
}


struct Quaternion {

  double a;
  double b;
  double c;
  double d;

  Quaternion() {}
  
  Quaternion(double a_, double b_, double c_, double d_)
    : a(a_), b(b_), c(c_), d(d_) {}
  
  Quaternion(double* p)
    : a(p[0]), b(p[1]), c(p[2]), d(p[3]) {}
  
  inline double operator[] (long k) {
    return (*this)[k];
  }
  inline const double operator[] (long k) const {
    return (*this)[k];
  }
  inline Quaternion& operator= (const Quaternion& r) {
    a = r.a;
    b = r.b;
    c = r.c;
    d = r.d;
    return (*this);
  }
  inline Quaternion operator+ (const Quaternion& r) const {
    return Quaternion(a + r.a, b + r.b, c + r.c, d + r.d);
  }
  inline Quaternion operator- () const {
    return Quaternion(-a, -b, -c, -d);
  }
  inline Quaternion operator- (const Quaternion& r) const {
    return Quaternion(a - r.a, b - r.b, c - r.c, d - r.d);
  }
  /* Dot/scalar/inner product */
  inline double operator* (const Quaternion& r) const {
    return a * r.a + b * r.b + c * r.c + d * r.d;
  }
  inline Quaternion operator* (const double& alpha) const {
    return Quaternion(alpha * a, alpha * b, alpha * c, alpha * d);
  }
  inline Quaternion operator/ (const double& alpha) const {
    return (*this) * (1.0 / alpha);
  }
  inline double norm () const {
    return sqrtf(a * a + b * b + c * c + d * d);
  }
  /* Hadamard/Schur/entrywise product */
  inline Quaternion operator& (const Quaternion& r) const {
    return Quaternion(a * r.a, b * r.b, c * r.c, d * r.d);
  }
  
};

inline Quaternion operator* (const double& alpha, const Quaternion& q) {
  return q * alpha;
}
#endif /* VECTORS_H */
