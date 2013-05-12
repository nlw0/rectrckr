#ifndef CAMERA_MODELS_H
#define CAMERA_MODELS_H
#include "vectors.h"

enum CameraModelType {
  PERSPECTIVE_PROJ=1,
  HARRIS_PROJ,
  EQUIDISTANT_PROJ,
  EQUIRECTANGULAR_PROJ
};

struct CameraModel {
  CameraModelType model; /* Which model*/
  double cx; /* origin, or projection center coordinates (principal point) */
  double cy;
  double fl; /* focal length */
  double kappa; /* wildcard distortion parameter */
};

/* Forward projections */
Vector2D project_perspective(const Vector3D& q, const CameraModel& cam);
Vector2D project_harris(const Vector3D& q, const CameraModel& cam);
Vector2D project_equidistant(const Vector3D& q, const CameraModel& cam);
Vector2D project_equirectangular(const Vector3D& q, const CameraModel& cam);
/* Inverse projections */
Vector3D invproject_perspective(const Vector2D& p, const CameraModel& cam);
Vector3D invproject_harris(const Vector2D& p, const CameraModel& cam);
Vector3D invproject_equidistant(const Vector3D& p, const CameraModel& cam);
Vector3D invproject_equirectangular(const Vector2D& p, const CameraModel& cam);

Vector2D project(const Vector3D& q, const CameraModel& cam);
Vector2D invproject(const Vector3D& q, const CameraModel& cam);
#endif /* CAMERA_MODELS_H */
