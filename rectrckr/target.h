#include "vectors.h"
#include "camera_models.h"

static inline Vector2D p(const Vector3D& s, const Vector3D& t, const Quaternion& psi);
static inline Vector3D q(const Vector3D& s, const Vector3D& t, const Quaternion& psi);

static inline Vector3D dq_ds(const Quaternion& psi, const int& k);
static inline Vector3D dq_dt(const Quaternion& psi, const int& k);

static inline Vector3D dq_dpsi(const Vector3D& s, const Vector3D& t, const Quaternion& psi, const int& k);

/* Reference frame vector at the specified direction. */
static inline Vector3D r(Quaternion psi, int direction);

/* Derivative of each vector component relative to the quaternion
   component k. */
static inline Vector3D dr_dpsi(Quaternion psi, int direction, int k);
