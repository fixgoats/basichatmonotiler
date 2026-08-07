#ifndef BASICHATMONOTILE_AFFINE_H
#define BASICHATMONOTILE_AFFINE_H

#include "Eigen/Dense"
#include "typedefs.h"

using Eigen::Vector2d, Eigen::Matrix3d, Eigen::Matrix3Xd, Eigen::Vector3d,
    Eigen::Matrix2d, Eigen::indexing::all, Eigen::MatrixX3d, Eigen::Matrix2Xd,
    Eigen::MatrixX2d, std::numbers::sqrt3, std::numbers::pi;

[[nodiscard]] constexpr Matrix3d sixth_rot(u64 i) noexcept {
  switch (i) {
  case 0:
    return Matrix3d::Identity();
  case 1:
    return (Matrix3d() << 0.5, -sqrt3 / 2., 0., sqrt3 / 2., 0.5, 0., 0., 0., 1.)
        .finished();
  case 2:
    return (Matrix3d() << -0.5, -sqrt3 / 2., 0., sqrt3 / 2., -0.5, 0., 0., 0.,
            1.)
        .finished();
  case 3:
    return (Matrix3d() << -1., 0., 0., 0., -1., 0., 0., 0., 1.).finished();
  case 4:
    return (Matrix3d() << -0.5, sqrt3 / 2., 0., -sqrt3 / 2., -0.5, 0., 0., 0.,
            1.)
        .finished();
  case 5:
    return (Matrix3d() << 0.5, sqrt3 / 2., 0., -sqrt3 / 2., 0.5, 0., 0., 0., 1.)
        .finished();
  default:
    return sixth_rot(i % 6);
  }
}

[[nodiscard]] constexpr Matrix3d affrot(f64 ang) noexcept {
  return (Matrix3d() << std::cos(ang), -std::sin(ang), 0, std::sin(ang),
          std::cos(ang), 0., 0., 0., 1.)
      .finished();
}

[[nodiscard]] constexpr Matrix3d transl2(Vector2d v) noexcept {
  return (Matrix3d() << 1., 0., v[0], 0., 1., v[1], 0., 0., 1.).finished();
}

[[nodiscard]] constexpr Matrix3d transl3(Vector3d v) noexcept {
  return (Matrix3d() << 1., 0., v[0], 0., 1., v[1], 0., 0., 1.).finished();
}

[[nodiscard]] constexpr Matrix3d rot_about(Vector3d v, u64 ang) noexcept {
  return Matrix3d{transl3(v) * (sixth_rot(ang) * transl3(-v))};
}

[[nodiscard]] constexpr Matrix3d from_seg(Vector3d p, Vector3d q) noexcept {
  return (Matrix3d() << q[0] - p[0], p[1] - q[1], p[0], q[1] - p[1],
          q[0] - p[0], p[1], 0., 0., 1.)
      .finished();
}

[[nodiscard]] constexpr Matrix3d aff_inv(Matrix3d aff) noexcept {
  Matrix2d mat_part =
      (Matrix2d() << aff(0, 0), aff(0, 1), aff(1, 0), aff(1, 1)).finished();
  Matrix2d mat_part_inv = mat_part.inverse();
  Vector2d transl_part = -mat_part_inv * Vector2d{aff(0, 2), aff(1, 2)};
  return (Matrix3d() << mat_part_inv(0, 0), mat_part_inv(0, 1), transl_part[0],
          mat_part_inv(1, 0), mat_part_inv(1, 1), transl_part[1], 0., 0., 1.)
      .finished();
}

[[nodiscard]] constexpr Matrix3d match_segs(Vector3d p1, Vector3d q1,
                                            Vector3d p2, Vector3d q2) noexcept {
  return Matrix3d{from_seg(p2, q2) * (aff_inv(from_seg(p1, q1)))};
}

[[nodiscard]] constexpr Matrix3d translate_by(Matrix3d aff,
                                              Vector2d v) noexcept {
  aff(0, 2) += v[0];
  aff(1, 2) += v[1];
  return aff;
}

[[nodiscard]] constexpr Matrix3d translate_by3(Matrix3d aff,
                                               Vector3d v) noexcept {
  aff(0, 2) += v[0];
  aff(1, 2) += v[1];
  return aff;
}

[[nodiscard]] constexpr Matrix3d reflect_y(Matrix3d aff) noexcept {
  Matrix3d ret = aff;
  ret(0, 0) *= -1;
  ret(0, 1) *= -1;
  return aff;
}

[[nodiscard]] constexpr Vector3d scale(Vector3d v, f64 a) noexcept {
  return {v.x() * a, v.y() * a, 1.0};
}

[[nodiscard]] constexpr Vector3d
intersection(Vector3d p1, Vector3d q1, Vector3d p2, Vector3d q2) noexcept {
  const f64 d =
      (q2[1] - p2[1]) * (q1[0] - p1[0]) - (q2[0] - p2[0]) * (q1[1] - p1[1]);
  const f64 u_a = ((q2.x() - p2.x()) * (p1.y() - p2.y()) -
                   (q2.y() - p2.y()) * (p1.x() - p2.x())) /
                  d;
  return (Vector3d() << p1.x() + u_a * (q1.x() - p1.x()),
          p1.y() + u_a * (q1.y() - p1.y()), 1.)
      .finished();
}

[[nodiscard]] constexpr Vector3d affsub(Vector3d v, Vector3d u) noexcept {
  return Vector3d{v[0] - u[0], v[1] - u[1], 1.};
}

[[nodiscard]] constexpr Vector3d affadd(Vector3d v, Vector3d u) noexcept {
  return Vector3d{v[0] + u[0], v[1] + u[1], 1.};
}

#endif // BASICHATMONOTILE_AFFINE_H
