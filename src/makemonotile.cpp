#include "Eigen/Dense"
#include "kdtree.h"
#if MONOTILE_VISUAL
#include "raylib.h"
#endif // MONOTILE_VISUAL
#include "typedefs.h"
#include <cmath>
#include <cstddef>
#include <cxxopts.hpp>
#include <exception>
#include <format>
#include <fstream>
#include <iostream>
#include <numbers>
#include <numeric>
#include <variant>
#include <vector>

using Eigen::Vector2d, Eigen::Matrix3d, Eigen::Matrix3Xd, Eigen::Vector3d,
    Eigen::Matrix2d, Eigen::indexing::all, Eigen::MatrixX3d, Eigen::Matrix2Xd,
    Eigen::MatrixX2d, std::numbers::sqrt3;

static const Eigen::Matrix<f64, 3, 13> HAT =
    (Eigen::Matrix<f64, 3, 13>() << 0., -0.75, -0.5, 0.5, 0.75, 1.5, 2.25, 2.,
     1.5, 1.5, 0.75, 0.5, 0., 0., -0.25 * sqrt3, -0.5 * sqrt3, -0.5 * sqrt3,
     -0.25 * sqrt3, -0.5 * sqrt3, -0.25 * sqrt3, 0., 0., 0.5 * sqrt3,
     0.75 * sqrt3, 0.5 * sqrt3, 0.5 * sqrt3, 1., 1., 1., 1., 1., 1., 1., 1., 1.,
     1., 1., 1., 1.)
        .finished();

static const Eigen::Matrix<f64, 3, 13> HAT1 =
    (Eigen::Matrix<f64, 3, 13>() << 0., 0.75, 0.5, -0.5, -0.75, -1.5, -2.25,
     -2., -1.5, -1.5, -0.75, -0.5, 0., 0., -0.25 * sqrt3, -0.5 * sqrt3,
     -0.5 * sqrt3, -0.25 * sqrt3, -0.5 * sqrt3, -0.25 * sqrt3, 0., 0.,
     0.5 * sqrt3, 0.75 * sqrt3, 0.5 * sqrt3, 0.5 * sqrt3, 1., 1., 1., 1., 1.,
     1., 1., 1., 1., 1., 1., 1., 1.)
        .finished();

struct Pt2 : std::array<f64, 2> {
  static constexpr size_t DIM = 2;
};

static const std::vector<std::vector<size_t>> RULES{{0},
                                                    {2, 0, 0, 2},
                                                    {0, 1, 0, 2},
                                                    {2, 2, 0, 2},
                                                    {0, 3, 0, 2},
                                                    {2, 4, 4, 2},
                                                    {3, 0, 4, 3},
                                                    {3, 2, 4, 3},
                                                    {3, 4, 1, 3, 2, 0},
                                                    {0, 8, 3, 0},
                                                    {2, 9, 2, 0},
                                                    {0, 10, 2, 0},
                                                    {2, 11, 4, 2},
                                                    {0, 12, 0, 2},
                                                    {3, 13, 0, 3},
                                                    {3, 14, 2, 1},
                                                    {0, 15, 3, 4},
                                                    {3, 8, 2, 1},
                                                    {0, 17, 3, 0},
                                                    {2, 18, 2, 0},
                                                    {0, 19, 2, 2},
                                                    {3, 20, 4, 3},
                                                    {2, 20, 0, 2},
                                                    {0, 22, 0, 2},
                                                    {3, 23, 4, 3},
                                                    {3, 23, 0, 3},
                                                    {2, 16, 0, 2},
                                                    {1, 9, 4, 0, 2, 2},
                                                    {3, 4, 0, 3}};

namespace {
constexpr Matrix3d sixth_rot(u64 i) {
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

constexpr Matrix3d transl2(Vector2d v) {
  return (Matrix3d() << 1., 0., v[0], 0., 1., v[1], 0., 0., 1.).finished();
}

constexpr Matrix3d transl3(Vector3d v) {
  return (Matrix3d() << 1., 0., v[0], 0., 1., v[1], 0., 0., 1.).finished();
}

constexpr Matrix3d rot_about(Vector3d v, u64 ang) {
  return Matrix3d{transl3(v) * (sixth_rot(ang) * transl3(-v))};
}

constexpr Matrix3d from_seg(Vector3d p, Vector3d q) {
  return (Matrix3d() << q[0] - p[0], p[1] - q[1], p[0], q[1] - p[1],
          q[0] - p[0], p[1], 0., 0., 1.)
      .finished();
}

constexpr Matrix3d aff_inv(Matrix3d aff) {
  Matrix2d mat_part =
      (Matrix2d() << aff(0, 0), aff(0, 1), aff(1, 0), aff(1, 1)).finished();
  Matrix2d mat_part_inv = mat_part.inverse();
  Vector2d transl_part = -mat_part_inv * Vector2d{aff(0, 2), aff(1, 2)};
  return (Matrix3d() << mat_part_inv(0, 0), mat_part_inv(0, 1), transl_part[0],
          mat_part_inv(1, 0), mat_part_inv(1, 1), transl_part[1], 0., 0., 1.)
      .finished();
}

constexpr Matrix3d match_segs(Vector3d p1, Vector3d q1, Vector3d p2,
                              Vector3d q2) {
  return Matrix3d{from_seg(p2, q2) * (aff_inv(from_seg(p1, q1)))};
}

constexpr Matrix3d translate_by(Matrix3d aff, Vector2d v) {
  aff(0, 2) += v[0];
  aff(1, 2) += v[1];
  return aff;
}

Vector3d intersection(Vector3d p1, Vector3d q1, Vector3d p2, Vector3d q2) {
  const f64 d =
      (q2[1] - p2[1]) * (q1[0] - p1[0]) - (q2[0] - p2[0]) * (q1[1] - p1[1]);
  const f64 u_a = ((q2.x() - p2.x()) * (p1.y() - p2.y()) -
                   (q2.y() - p2.y()) * (p1.x() - p2.x())) /
                  d;
  return (Vector3d() << p1.x() + u_a * (q1.x() - p1.x()),
          p1.y() + u_a * (q1.y() - p1.y()), 1.)
      .finished();
}

constexpr Vector3d affsub(Vector3d v, Vector3d u) {
  return Vector3d{v[0] - u[0], v[1] - u[1], 1.};
}

constexpr Vector3d affadd(Vector3d v, Vector3d u) {
  return Vector3d{v[0] + u[0], v[1] + u[1], 1.};
}

template <class... Ts>
struct Overloads : Ts... {
  using Ts::operator()...;
};

typedef std::variant<Eigen::Matrix<f64, 3, 3>, Eigen::Matrix<f64, 3, 4>,
                     Eigen::Matrix<f64, 3, 5>, Eigen::Matrix<f64, 3, 6>>
    shape_var_t;
struct ShapeMaker {
  Matrix3d transform;
  std::shared_ptr<shape_var_t> p_shape;

  [[nodiscard]] s32 shape_len() const {
    return std::visit<s32>([](auto a) { return a.cols(); }, *p_shape);
  }
  template <class Int>
  [[nodiscard]] Vector3d eval_shape_pt(Int i) const {
    //

    Vector3d ret = std::visit<Vector3d>(
        [this, i](auto shape) {
          return Vector3d{transform * shape(all, static_cast<s64>(i))};
        },
        *p_shape);
    //
    return ret;
  }
  template <class Int>
  [[nodiscard]] Vector3d shape_pt(Int i) const {
    return std::visit<Vector3d>(
        [i](auto shape) { return Vector3d{shape(all, static_cast<s64>(i))}; },
        *p_shape);
  }
  [[nodiscard]] Matrix3Xd to_hats(Matrix3d transf) const {
    const auto visitor = Overloads{
        [this, transf](Eigen::Matrix<f64, 3, 6>) {
          //
          return Matrix3Xd{
              transf * transform *
              (Matrix3Xd(3, 4 * 13)
                   << translate_by(sixth_rot(5), {2.5, 0.5 * sqrt3}) * HAT1,
               translate_by(sixth_rot(4), {1., sqrt3}) * HAT,
               translate_by(sixth_rot(4), {4., sqrt3}) * HAT,
               translate_by(sixth_rot(2), {2.5, 1.5 * sqrt3}) * HAT)
                  .finished()};
        },
        [this, transf](Eigen::Matrix<f64, 3, 3>) {
          //
          return Matrix3Xd{transf * transform * transl2({0.5, 0.5 * sqrt3}) *
                           HAT};
        },
        [this, transf](Eigen::Matrix<f64, 3, 4>) {
          //
          return Matrix3Xd{
              transf * transform *
              (Matrix3Xd(3, 2 * 13) << transl2({1.5, 0.5 * sqrt3}) * HAT,
               translate_by(sixth_rot(5), {0., sqrt3}) * HAT)
                  .finished()};
        },
        [this, transf](Eigen::Matrix<f64, 3, 5>) {
          //
          return Matrix3Xd{
              transf * transform *
              (Matrix3Xd(3, 2 * 13) << transl2({1.5, 0.5 * sqrt3}) * HAT,
               translate_by(sixth_rot(5), {0., sqrt3}) * HAT)
                  .finished()};
        }};
    return std::visit<Matrix3Xd>(visitor, *p_shape); //->visit(visitor);
  }
};

template <class T>
struct Node {
  std::vector<std::shared_ptr<Node<T>>> children;
  T data{};

  Node(T obj) : data{obj} {
    //
  }
  Node(const std::vector<std::shared_ptr<Node<T>>>& ch, T obj)
      : children{ch}, data{std::move(obj)} {}
};

template <class T>
struct Tree {
  std::shared_ptr<Node<T>> root;
  Tree<T>(Node<T>* r) : root{r} {
    //
  }
  Tree<T>(std::shared_ptr<Node<T>> r) : root{r} {}
  Tree<T>(T r) : root{std::make_shared<Node<T>>(r)} {}
};

void nodes_to_hats(const Node<ShapeMaker>& nd, Matrix3d transf,
                   std::vector<Eigen::Matrix3Xd>& hats) {
  if (nd.children.size() == 0) {
    auto hat_pts = nd.data.to_hats(transf);
    hats.push_back(hat_pts);
  } else {
    for (const auto& ch : nd.children) {
      nodes_to_hats(*ch, transf * nd.data.transform, hats);
    }
  }
}

Eigen::Matrix3Xd tree_to_hats(const Tree<ShapeMaker>& tree,
                              size_t init_size = 1000) {
  std::vector<Eigen::Matrix3Xd> hats;
  hats.reserve(init_size);
  nodes_to_hats(*tree.root, Matrix3d::Identity(), hats);
  u64 n_cols = 0;
  for (const auto& pts : hats) {
    n_cols += pts.cols();
  }
  Matrix3Xd all_hats(3, n_cols);
  u64 cur_col = 0;
  for (const auto& pts : hats) {
    all_hats(all, Eigen::seqN(cur_col, pts.cols())) = pts;
    cur_col += pts.cols();
  }
  return all_hats;
}

void iter_trees(Tree<ShapeMaker>* trees) {
  std::array<std::shared_ptr<Node<ShapeMaker>>, 29> patch;
  size_t ch_counter = 0;
  for (const auto& r : RULES) {
    if (r.size() == 1) {
      auto nshp = std::make_shared<Node<ShapeMaker>>(*trees[0].root);
      nshp->data.transform = Matrix3d::Identity();
      patch[ch_counter] = nshp;
      ++ch_counter;
    } else if (r.size() == 4) {

      const Vector3d p = patch[r[1]]->data.eval_shape_pt(
          (r[2] + 1) % patch[r[1]]->data.shape_len());

      const Vector3d q = patch[r[1]]->data.eval_shape_pt(r[2]);

      auto nshp = std::make_shared<Node<ShapeMaker>>(*trees[r[0]].root);
      Vector3d pt1 = nshp->data.shape_pt(r[3]);

      Vector3d pt2 = nshp->data.shape_pt((r[3] + 1) % nshp->data.shape_len());

      const Matrix3d match_transf = match_segs(pt1, pt2, p, q);
      nshp->data.transform = match_transf;
      patch[ch_counter] = nshp;
      ++ch_counter;
    } else {

      const Vector3d p = patch[r[3]]->data.eval_shape_pt(r[4]);

      const Vector3d q = patch[r[1]]->data.eval_shape_pt(r[2]);

      auto nshp = std::make_shared<Node<ShapeMaker>>(*trees[r[0]].root);
      Vector3d pt1 = nshp->data.shape_pt(r[5]);

      Vector3d pt2 = nshp->data.shape_pt((r[5] + 1) % nshp->data.shape_len());

      const Matrix3d match_trans = match_segs(pt1, pt2, p, q);
      nshp->data.transform = match_trans;
      patch[ch_counter] = nshp;
      ++ch_counter;
    }
  }

  const Vector3d bps1 = patch[8]->data.eval_shape_pt(2);
  const Vector3d bps2 = patch[21]->data.eval_shape_pt(2);
  const Vector3d rbps = rot_about(bps1, 4) * bps2;
  const Vector3d p72 = patch[7]->data.eval_shape_pt(2);
  const Vector3d p252 = patch[25]->data.eval_shape_pt(2);
  const Vector3d llc =
      intersection(bps1, rbps, patch[6]->data.eval_shape_pt(2), p72);

  auto w = affsub(patch[6]->data.eval_shape_pt(2), llc);

  Eigen::Matrix<f64, 3, 6> new_h_outline{};
  new_h_outline(all, 0) = llc;
  new_h_outline(all, 1) = bps1;
  w = sixth_rot(5) * w;
  new_h_outline(all, 2) = affadd(new_h_outline(all, 1), w);
  new_h_outline(all, 3) = patch[14]->data.eval_shape_pt(2);
  w = sixth_rot(5) * w;
  new_h_outline(all, 4) = affsub(new_h_outline(all, 3), w);
  new_h_outline(all, 5) = patch[6]->data.eval_shape_pt(2);
  std::vector<std::shared_ptr<Node<ShapeMaker>>> a{
      patch[0], patch[9], patch[16], patch[27], patch[26],
      patch[6], patch[1], patch[8],  patch[10], patch[15]};
  trees[0].root = std::make_shared<Node<ShapeMaker>>(Node<ShapeMaker>{
      {patch[0], patch[9], patch[16], patch[27], patch[26], patch[6], patch[1],
       patch[8], patch[10], patch[15]},
      {.transform = Matrix3d::Identity(),
       .p_shape = std::make_shared<shape_var_t>(new_h_outline)}});

  const Vector3d aaa = new_h_outline(all, 2);
  const Vector3d bbb =
      affadd(new_h_outline(all, 1),
             affsub(new_h_outline(all, 4), new_h_outline(all, 5)));
  const Vector3d ccc = rot_about(bbb, 5) * aaa;
  Matrix3d t_shape;
  t_shape(all, 0) = bbb;
  t_shape(all, 1) = ccc;
  t_shape(all, 2) = aaa;
  trees[1].root = std::make_shared<Node<ShapeMaker>>(
      Node<ShapeMaker>{{patch[11]},
                       {.transform = Matrix3d::Identity(),
                        .p_shape = std::make_shared<shape_var_t>(t_shape)}

      });

  Eigen::Matrix<f64, 3, 4> p_shape;
  p_shape(all, 0) = p72;
  p_shape(all, 1) = affadd(p72, affsub(bps1, llc));
  p_shape(all, 2) = bps1;
  p_shape(all, 3) = llc;
  trees[2].root = std::make_shared<Node<ShapeMaker>>(
      Node<ShapeMaker>{{patch[7], patch[2], patch[3], patch[4], patch[28]},
                       {.transform = Matrix3d::Identity(),
                        .p_shape = std::make_shared<shape_var_t>(p_shape)}}

  );

  Eigen::Matrix<f64, 3, 5> f_shape;
  f_shape(all, 0) = bps2;
  f_shape(all, 1) = patch[24]->data.eval_shape_pt(2);
  f_shape(all, 2) = patch[25]->data.eval_shape_pt(0);
  f_shape(all, 3) = p252;
  f_shape(all, 4) = affadd(p252, affsub(llc, bps1));
  trees[3].root = std::make_shared<Node<ShapeMaker>>(Node<ShapeMaker>{
      {patch[21], patch[20], patch[22], patch[23], patch[24], patch[25]},
      {.transform = Matrix3d::Identity(),
       .p_shape = std::make_shared<shape_var_t>(f_shape)}});
}

s32 to_screen_isotropic(f64 r, f64 start, f64 scale, s32 dim) {
  return (s32)(((r - start) / scale) * (f64)dim);
}

Matrix2Xd filterpts(const Matrix3Xd& pts) {
  std::vector<u8> uniques(pts.cols(), 1);

  kdt::KDTree<Pt2> kdtree([](const Matrix3Xd& pts) {
    std::vector<Pt2> kdpts(pts.cols());
#pragma omp parallel for
    for (s32 i = 0; i < pts.cols(); ++i) {
      kdpts[i] = {pts(0, i), pts(1, i)};
    }
    return kdpts;
  }(pts));

  for (s32 i = 0; i < pts.cols(); ++i) {
    if (static_cast<bool>(uniques[i])) {
      const auto dupes = kdtree.radiusSearch(kdtree.points_[i], 1e-5);
      for (const auto& idx : dupes) {
        if (idx > i) {
          uniques[idx] = 0;
        }
      }
    }
  }
  u64 num_uniques = std::accumulate(uniques.begin(), uniques.end(), 0);
  Eigen::Matrix2Xd unique_pts(2, num_uniques);
  s64 count = 0;
  for (s64 i = 0; i < pts.cols(); ++i) {
    if (static_cast<bool>(uniques[i])) {
      unique_pts(all, count) = pts(Eigen::seq(0, 1), i);
      ++count;
    }
  }
  return unique_pts;
}

} // namespace
int main(int argc, char* argv[]) {
  cxxopts::Options options("makemonotile",
                           "Create a section of the hat monotile tiling, based "
                           "on the code developed by Craig S. Kaplan et al. "
                           "at: https://github.com/isohedral/hatviz");

  options.add_options()("h,help", "Show this help text")(
      "l,level", "Number of metatile iterations", cxxopts::value<u64>())(
      "w,window", "Draw generated section of tiling"
#ifndef MONOTILE_VISUAL
                  ". Disabled, compile with -DMONOTILE_VISUAL to enable"
#endif // !MONOTILE_VISUAL
      )("o,output", "Point output",
        cxxopts::value<std::string>()->default_value("monopts.txt"));
  cxxopts::ParseResult result;

  try {
    result = options.parse(argc, argv);
  } catch (const std::exception& exc) {
    std::cout << options.help() << std::endl;
    return EXIT_FAILURE;
  }

  if (static_cast<bool>(result["h"].count())) {
    std::cout << options.help() << std::endl;
    exit(0);
  }
  u64 n_iters = 0;
  if (static_cast<bool>(result["l"].count())) {
    n_iters = result["l"].as<u64>();
  }

  std::vector<Tree<ShapeMaker>> tiles = {
      ShapeMaker{.transform = Matrix3d::Identity(),
                 .p_shape = std::make_shared<shape_var_t>(
                     (Eigen::Matrix<f64, 3, 6>() << 0., 4., 4.5, 2.5, 1.5, -0.5,
                      0., 0., 0.5 * sqrt3, 2.5 * sqrt3, 2.5 * sqrt3,
                      0.5 * sqrt3, 1., 1., 1., 1., 1., 1.)
                         .finished())},
      ShapeMaker{

          .transform = Matrix3d::Identity(),
          .p_shape = std::make_shared<shape_var_t>(Eigen::Matrix3d{
              {0., 3., 1.5}, {0., 0., 1.5 * sqrt3}, {1., 1., 1.}}),
      },
      ShapeMaker{
          .transform = Matrix3d::Identity(),
          .p_shape = std::make_shared<shape_var_t>(
              (Eigen::Matrix<f64, 3, 4>() << 0., 4., 3., -1., 0., 0., sqrt3,
               sqrt3, 1., 1., 1., 1.)
                  .finished()),
      },
      ShapeMaker{
          .transform = Matrix3d::Identity(),
          .p_shape = std::make_shared<shape_var_t>(
              (Eigen::Matrix<f64, 3, 5>() << 0., 3., 3.5, 3., -1., 0., 0.,
               0.5 * sqrt3, sqrt3, sqrt3, 1., 1., 1., 1., 1.)
                  .finished()),
      }};
  for (u32 i = 0; i < n_iters; ++i) {
    iter_trees(tiles.data());
  }

  const auto points = tree_to_hats(tiles[0]);
  if (result["w"].as<bool>()) {
#ifndef MONOTILE_VISUAL
    std::cout
        << "Executable is not built with raylib, so no output is shown. This "
           "is the default since raylib is a fairly large download.\n"
           "To enable this functionality, pass in -DMONOTILE_VISUAL=ON when"
           " configuring the build directory\n";
#else
    s32 width = 800;
    s32 height = 800;
    SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_WINDOW_RESIZABLE |
                   FLAG_WINDOW_TRANSPARENT);
    InitWindow(width, height, "raylib test");

    SetTargetFPS(10);
    std::cout << "number of points: " << points.cols() << '\n';
    f64 xmin = points(0, all).minCoeff();
    f64 xmax = points(0, all).maxCoeff();
    f64 ymin = points(1, all).minCoeff();
    f64 ymax = points(1, all).maxCoeff();
    f64 exmin = xmin - 0.05 * (xmax - xmin);
    f64 eymin = ymin - 0.05 * (ymax - ymin);
    f64 exmax = xmax + 0.05 * (xmax - xmin);
    f64 eymax = ymax + 0.05 * (ymax - ymin);
    f64 max_of_exey = std::max(eymax - eymin, exmax - exmin);
    while (!WindowShouldClose()) {
      width = GetScreenWidth();
      height = GetScreenHeight();
      s32 min_of_wh = std::min(width, height);
      BeginDrawing();
      ClearBackground(WHITE);
      for (s32 i = 0; i < points.cols() / 13; ++i) {
        for (s32 j = 0; j < 13; ++j) {
          DrawLineEx(
              {static_cast<f32>(to_screen_isotropic(
                   points(0, i * 13 + j), exmin, max_of_exey, min_of_wh)),
               static_cast<f32>(to_screen_isotropic(
                   points(1, i * 13 + j), eymin, max_of_exey, min_of_wh))},
              {static_cast<f32>(
                   to_screen_isotropic(points(0, i * 13 + (j + 1) % 13), exmin,
                                       max_of_exey, min_of_wh)),
               static_cast<f32>(
                   to_screen_isotropic(points(1, i * 13 + (j + 1) % 13), eymin,
                                       max_of_exey, min_of_wh))},
              4.0, BLACK);
        }
      }

      EndDrawing();
    }
    CloseWindow();
#endif // MONOTILE_VISUAL
  }
  Matrix2Xd unique_pts = filterpts(points);
  std::string fname = result["o"].as<std::string>();
  std::ofstream outfile(fname);

  for (s64 i = 0; i < unique_pts.cols(); ++i) {
    outfile << std::format("{}", unique_pts(0, i)) << ' '
            << std::format("{}", unique_pts(1, i)) << '\n';
  }
  outfile.close();
  std::cout << "Wrote " << unique_pts.cols() << " points to file: " << fname
            << '\n';
  return 0;
}
