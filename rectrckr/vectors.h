#include <math.h>

struct Vector2D {

  double x;
  double y;

  Vector2D() {}

  Vector2D(double x_, double y_)
  : x(x_), y(y_) {}

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

};

static inline double Vector3D::operator[] (Vector3D& a) {
  return (&a)[k];
}
static inline const double Vector3D::operator[] (const Vector3D& a) {
  return (&a)[k];
}
static inline Vector3D& Vector3D::operator= (Vector3D& a, const Vector3D& b) {
  a.x = b.x;
  a.y = b.y;
  a.z = b.z;
  return a;
}
static inline Vector3D Vector3D::operator+ (const Vector3D& a, const Vector3D& b) {
  return Vector3D(a.x + b.x, a.y + b.y, a.z + b.z);
}
static inline Vector3D Vector3D::operator- (const Vector3D& b) {
  return Vector3D(-b.x, -b.y, -b.z);
}
static inline Vector3D Vector3D::operator- (const Vector3D& a, const Vector3D& b) {
  return Vector3D(a.x - b.x, a.y - b.y, a.z - b.z);
}
/* Dot/scalar/inner product */
static inline double Vector3D::operator* (const Vector3D& a, const Vector3D& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
static inline Vector3D Vector3D::operator* (const double& alpha, const Vector3D& a) {
  return Vector3D(alpha * a.x, alpha * a.y, alpha * a.z);
}
static inline Vector3D Vector3D::operator* (const Vector3D& a, const double& alpha) {
  return alpha * a;
}
static inline Vector3D Vector3D::operator/ (const Vector3D& a, const double& alpha) {
  return a * (1.0 / alpha);
}
static inline Vector3D Vector3D::norm (const Vector3D& a) {
  return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
}
/* Cross/vector product */
static inline Vector3D Vector3D::operator% (const Vector3D& a, const Vector3D& b) {
  return Vector3D(a.y * b.z - a.z * b.y,
                  a.z * b.x - a.x * b.z,
                  a.x * b.y - a.y * b.x);
}
/* Hadamard/Schur/entrywise product */
static inline Vector3D Vector3D::operator& (const Vector3D& a, const Vector3D& b) {
  return Vector3D(a.x * b.x, a.y * b.y, a.z * b.z);
}


struct Quaternion {

  Quaternion(double a_, double b_, double c_, double d_)
    : a(a_), b(b_), c(c_), d(d_) {}
  
  double a;
  double b;
  double c;
  double d;

  Quaternion() {}

  Quaternion(double a_, double b_, double c_, double d_)
  : a(a_), b(b_), c(c_), d(d_) {}

};

static inline double Quaternion::operator[] (Quaternion& a) {
  return (&a)[k];
}
static inline const double Quaternion::operator[] (const Quaternion& a) {
  return (&a)[k];
}
static inline Quaternion& Quaternion::operator= (Quaternion& q, const Quaternion& r) {
  q.a = r.a;
  q.b = r.b;
  q.c = r.c;
  q.d = r.d;
  return q;
}
static inline Quaternion Quaternion::operator+ (const Quaternion& q, const Quaternion& r) {
  return Quaternion(q.a + r.a, q.b + r.b, q.c + r.c, q.d + r.d);
}
static inline Quaternion Quaternion::operator- (const Quaternion& q) {
  return Quaternion(-q.a, -q.b, -q.c, -q.d);
}
static inline Quaternion Quaternion::operator- (const Quaternion& q, const Quaternion& r) {
  return Quaternion(q.a - r.a, q.b - r.b, q.c - r.c, q.d - r.d);
}
/* Dot/scalar/inner product */
static inline double Quaternion::operator* (const Quaternion& q, const Quaternion& r) {
  return q.a * r.a + q.b * r.b + q.c * r.c + q.d * r.d;
}
static inline Quaternion Quaternion::operator* (const double& alpha, const Quaternion& q) {
  return Quaternion(alpha * q.a, alpha * q.b, alpha * q.c, alpha * q.d);
}
static inline Quaternion Quaternion::operator* (const Quaternion& q, const double& alpha) {
  return alpha * q;
}
static inline Quaternion Quaternion::operator/ (const Quaternion& q, const double& alpha) {
  return q * (1.0 / alpha);
}
static inline Quaternion Quaternion::norm (const Quaternion& q) {
  return sqrtf(q.a * q.a + q.b * q.b + q.c * q.c + q.d * q.d);
}
/* Hadamard/Schur/entrywise product */
static inline Quaternion Quaternion::operator& (const Quaternion& q, const Quaternion& r) {
  return Quaternion(q.a * r.a, q.b * r.b, q.c * r.c, q.d * r.d);
}
