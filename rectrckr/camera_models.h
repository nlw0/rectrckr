static inline Vector2D project_perspective(const Vector3D& q, double* parm) {
  return Vector2D(parm[1] + parm[0] * q.x / q.z,
                  parm[2] + parm[0] * q.x / q.z);
}

static inline Vector2D project_harris(const Vector3D& q, double* parm) {
  Vector2D p = project_perspective(q, parm);  
  double mul = rsqrt(fabs(1 -2 * parm[3] * p.norm()));
  return p * mul;
}

static inline Vector2D project_equidistant(const Vector3D& q, double* parm) {
  double phi = acos(q.z * q.inv_norm());
  double mul = parm[0] * phi * rsqrt(q.x * q.x + q.y * q.y);
  return Vector2D(parm[1] + mul * q.x,
                  parm[2] + mul * q.y);
}

static inline Vector2D project_equirectangular(const Vector3D& q, double* parm) {
  return Vector2D(parm[1] + parm[0] * atan2(q.z, q.x),
                  parm[2] + parm[0] * asin(q.y * q.inv_norm()));
}

static inline Vector2D project_polynomial(const Vector3D& q, double* parm) {
  Vector2D p = project_perspective(q, parm);
  double mul = 1 + parm[3] * (p.x * p.x + p.y * p.y);
  return p * mul;
}

static inline Vector2D project_fitzgibbon(const Vector3D& q, double* parm) {
  Vector2D p = project_perspective(q, parm);  
  double mul = 1.0 / fabs(1 -2 * parm[3] * (p.x * p.x + p.y * p.y));
  return p * mul;
}

/* Inverse projections */
static inline Vector3D invproject_perspective(const Vector2D& p, double* parm) {
  return Vector3D(p.x - parm[1], p.y - parm[2], parm[0]);
}

static inline Vector3D invproject_harris(const Vector2D& p, double* parm) {
  double mul = rsqrt(fabs(1 + 2 * parm[3] * p.norm()));
  return invproject_perspective(p * mul, parm);
}

static inline Vector3D invproject_equidistant(const Vector3D& p, double* parm) {
  Vector3D q(p.x - parm[1], p.y - parm[2], 0.0);
  double phi = q.norm();
  q = q / phi;
  phi = phi / parm[0];
  q = q * sin(phi);
  q.z = cos(phi);
  return q;
}

static inline Vector3D invproject_equirectangular(const Vector2D& p, double* parm) {
  double tht = (p.x - parm[1]) / parm[0];
  double phi = (p.y - parm[2]) / parm[0];

  double cos_phi = cos(phi);

  return Vector3D(cos_phi * sin(tht), sin(phi), cos_phi * cos(tht));
}
