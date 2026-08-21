//
// Created by fixgoats on 8/7/26.
#include "Eigen/Dense"
#include "affine.h"
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

typedef Eigen::Matrix<f64, 3, 4> Quad;
typedef Eigen::Matrix<f64, 3, 14> Tile;
// typedef std::variant<Quad, Tile> shape_var_t;

#ifndef NDEBUG
#define ASSERT(condition, message)                                             \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::cerr << "Assertion `" #condition "` failed in " << __FILE__         \
                << " line " << __LINE__ << ": " << message << std::endl;       \
      std::terminate();                                                        \
    }                                                                          \
  } while (false)
#else
#define ASSERT(condition, message)                                             \
  do {                                                                         \
  } while (false)
#endif

// inline void my_assert(bool condition, std::string message) {
//   if (!condition) {
//       std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
//                 << " line " << __LINE__ << ": " << message << std::endl; \
//       std::terminate(); \
//
//   }
// }

enum class Len : bool {
  a,
  b,
};

template <class T, size_t Cap>
struct SmallArr : std::array<T, Cap> {
  size_t size;

  using iter = const T*;
  using citer = const T*;

  constexpr SmallArr() = default;

  template <class... Args>
  consteval SmallArr(Args&&... args)
    requires(std::is_same_v<std::common_type_t<Args...>, T>)
      : std::array<T, Cap>{std::forward<Args>(args)...}, size{sizeof...(Args)} {
  }
  constexpr SmallArr(size_t s) : size{s}, std::array<T, Cap>{} {}

  [[nodiscard]] constexpr T operator[](auto i) const {
    ASSERT(i < size, "Attempted out of bounds access.");
    return this->data()[i];
  }
  [[nodiscard]] constexpr const T& operator[](auto i) {
    ASSERT(i < size, "Attempted out of bounds access.");
    return this->data()[i];
  }

  [[nodiscard]] T& operator[](auto i) {
    ASSERT(i < size, "Attempted out of bounds access.");
    return this->data()[i];
  }

  [[nodiscard]] T& front(auto i) {
    ASSERT(i < size, "Attempted out of bounds access.");
    return this->data()[0];
  }

  [[nodiscard]] T& back (auto i) {
    ASSERT(i < size, "Attempted out of bounds access.");
    return this->data()[size-1];
  }

  [[nodiscard]] constexpr T front(auto i) const {
    ASSERT(i < size, "Attempted out of bounds access.");
    return this->data()[0];
  }

  [[nodiscard]] constexpr T back () const {
    ASSERT(i < size, "Attempted out of bounds access.");
    return this->data()[size-1];
  }

  constexpr void push_back(T x) {
    ASSERT(size < Cap, "Pushing back would exceed capacity.");
    this->data()[size] = x;
    size += 1;
  }

  template <class... Args>
  constexpr void emplace_back(Args&&... args) {
    ASSERT(size < Cap, "Pushing back would exceed capacity.");
    this->data()[size] = T{std::forward<Args>(args)...};
    size += 1;
  }

  constexpr citer cbegin() const { return this->data(); }
  constexpr citer cend() const { return this->data() + size; }
  constexpr iter begin() const { return this->data(); }
  constexpr iter end() const { return this->data() + size; }
};

enum class Label : u8 {
  Delta = 0,
  Theta = 1,
  Lambda = 2,
  Xi = 3,
  Pi = 4,
  Sigma = 5,
  Phi = 6,
  Psi = 7,
  Gamma = 8,
};

constexpr std::array<Label, 9> LABELS{
    Label::Delta, Label::Theta, Label::Lambda, Label::Xi,    Label::Pi,
    Label::Sigma, Label::Phi,   Label::Psi,    Label::Gamma,
};
constexpr std::array<SmallArr<Label, 8>, 9> SUPER_RULES = {
    {{Label::Xi, Label::Delta, Label::Xi, Label::Phi, Label::Sigma, Label::Pi,
      Label::Phi, Label::Gamma},
     {Label::Psi, Label::Delta, Label::Pi, Label::Phi, Label::Sigma, Label::Pi,
      Label::Phi, Label::Gamma},
     {Label::Psi, Label::Delta, Label::Xi, Label::Phi, Label::Sigma, Label::Pi,
      Label::Phi, Label::Gamma},
     {Label::Psi, Label::Delta, Label::Pi, Label::Phi, Label::Sigma, Label::Psi,
      Label::Phi, Label::Gamma},
     {Label::Psi, Label::Delta, Label::Xi, Label::Phi, Label::Sigma, Label::Psi,
      Label::Phi, Label::Gamma},
     {Label::Xi, Label::Delta, Label::Xi, Label::Phi, Label::Sigma, Label::Pi,
      Label::Lambda, Label::Gamma},
     {Label::Psi, Label::Delta, Label::Psi, Label::Phi, Label::Sigma, Label::Pi,
      Label::Phi, Label::Gamma},
     {Label::Psi, Label::Delta, Label::Psi, Label::Phi, Label::Sigma,
      Label::Psi, Label::Phi, Label::Gamma},
     {Label::Pi, Label::Delta, Label::Theta, Label::Sigma, Label::Xi,
      Label::Phi, Label::Gamma}}};

;
// :
// ['Xi','Delta','Xi','Phi','Sigma','Pi','Phi','Gamma'],
// 		 		'Theta'
// :
// ['Psi','Delta','Pi','Phi','Sigma','Pi','Phi','Gamma'], 		'Lambda'
// :
// ['Psi','Delta','Xi','Phi','Sigma','Pi','Phi','Gamma'], 		'Xi' :
// ['Psi','Delta','Pi','Phi','Sigma','Psi','Phi','Gamma'], 		'Pi' :

struct Node {
  SmallArr<std::pair<std::shared_ptr<Node>, Matrix3d>, 8> children;
  std::shared_ptr<Quad> quad;
  // Label lab;

  Node() = default;
  // Node(Matrix3d tr, Quad* q, Label label)
  //     : transform{tr}, quad{q}, lab{label} {}
  Node(const SmallArr<std::pair<std::shared_ptr<Node>, Matrix3d>, 8>& ch,
       Quad* q)
      : children{ch}, quad{q} {}
};

struct Tree {
  std::shared_ptr<Node> root;
  Tree() = default;
  Tree(Node* r) : root{r} {
    //
  }
  Tree(std::shared_ptr<Node> r) : root{r} {}
  Tree(Node r) : root{std::make_shared<Node>(r)} {}
};

struct TNode {
  SmallArr<std::shared_ptr<TNode>, 6> children;
  Matrix3d transform;
  std::shared_ptr<Quad> quad;

  std::shared_ptr<TNode> rotate_and_match(Matrix3d t) {
    auto ret = std::make_shared<TNode>();
    ret->transform = t * transform;
    ret->quad = quad;
    return ret;
    // std::shared_ptr<Quad> = std::make_shared<Quad>()
  }
  // rotateAndMatch(T, qidx, P) {
  //   // First, construct a copy with all points transformed by the linear
  //   // operation T.
  //   const pts = this.pts.map(p = > transAB(T, p));
  //   const quad = this.quad.map(p = > transAB(T, p));
  //   const ret = new Shape(pts, quad, this.label);
  //   if (qidx >= 0) {
  //     ret.translateInPlace(psub(P, quad[qidx]));
  //   }
  //   return ret;
  // }
};

struct TTree {
  std::shared_ptr<TNode> root;
};

struct Rule {
  std::array<f64, 3> num;
  bool huh;
};

struct TRule {
  f64 ang;
  u32 i;
  u32 j;
  bool singcomp;
};

constexpr std::array<TRule, 7> T_RULES = {{
    {.ang = pi / 3, .i = 2, .j = 0, .singcomp = false},
    {.ang = 2 * pi / 3, .i = 2, .j = 0, .singcomp = false},
    {.ang = 0, .i = 1, .j = 1, .singcomp = true},
    {.ang = -2 * pi / 3, .i = 2, .j = 2, .singcomp = false},
    {.ang = -pi / 3, .i = 2, .j = 0, .singcomp = false},
    {.ang = 0, .i = 2, .j = 0, .singcomp = false},
}};

struct Pt2 : std::array<f64, 2> {
  static constexpr size_t DIM = 2;
};

constexpr f64 hsq3 = 0.5 * std::numbers::sqrt3;

static const std::array<Vector3d, 12> DIRS{{
    {1, 0, 1},
    {hsq3, 0.5, 1},
    {0.5, hsq3, 1},
    {0, 1, 1},
    {-0.5, hsq3, 1},
    {-hsq3, 0.5, 1},
    {-1, 0, 1},
    {-hsq3, -0.5, 1},
    {-0.5, -hsq3, 1},
    {0, -1, 1},
    {0.5, -hsq3, 1},
    {hsq3, -0.5, 1},
}};

struct Edge {
  u8 dir;
  Len len;

  [[nodiscard]] constexpr Vector3d vec(f64 a, f64 b) const {
    switch (len) {
    case Len::a: {
      return scale(DIRS[dir], a);
      break;
    }
    case Len::b: {
      return scale(DIRS[dir], b);
      break;
    }
    }
  }
};

constexpr std::array<Edge, 13> EDGES{{
    {.dir = 0, .len = Len::a},
    {.dir = 2, .len = Len::a},
    {.dir = 11, .len = Len::b},
    {.dir = 1, .len = Len::b},
    {.dir = 4, .len = Len::a},
    {.dir = 2, .len = Len::a},
    {.dir = 5, .len = Len::b},
    {.dir = 3, .len = Len::b},
    {.dir = 6, .len = Len::a},
    {.dir = 8, .len = Len::a},
    {.dir = 8, .len = Len::a},
    {.dir = 10, .len = Len::a},
    {.dir = 7, .len = Len::b},
}};

void iter_trees(std::array<Tree, 9> trees) {
  const Quad* ref = trees[static_cast<u8>(Label::Delta)].root->quad.get();
  f64 total_ang = 0;
  Matrix3d rot = Matrix3d::Identity();

  Quad tquad{};
  std::array<Matrix3d, 8> transforms{};
  transforms[0] = Matrix3d::Identity();
  for (u32 i = 0; i < 7; ++i) {
    total_ang += T_RULES[i].ang;
    if (T_RULES[i].ang != 0) {
      rot = affrot(total_ang);
      tquad = rot * (*ref);
    }
    const Vector3d ttt = affsub(transforms[i] * (*ref)(all, T_RULES[i].i),
                                tquad(all, T_RULES[i].j));
    transforms[i + 1] = translate_by3(rot, ttt);
  }
  for (auto& transform : transforms) {
    transform = reflect_y(transform);
  }

  auto super_quad = std::make_shared<Quad>();
  (*super_quad)(all, 0) = transforms[6] * (*ref)(all, 2);
  (*super_quad)(all, 1) = transforms[5] * (*ref)(all, 1);
  (*super_quad)(all, 2) = transforms[3] * (*ref)(all, 2);
  (*super_quad)(all, 3) = transforms[0] * (*ref)(all, 2);
  // transPt( Ts[6], quad[2] ),
  // transPt( Ts[5], quad[1] ),
  // transPt( Ts[3], quad[2] ),
  // transPt( Ts[0], quad[1] ) ];
  std::array<std::shared_ptr<Node>, 9> temp{};
  for (u32 i = 0; i < 9; ++i) {
    auto new_node = std::make_shared<Node>();
    for (u32 j = 0; j < SUPER_RULES[i].size; ++j) {
      new_node->children.push_back(
          {trees[static_cast<u8>(SUPER_RULES[i][j])].root, transforms[j]});
    }
    new_node->quad = super_quad;
    temp[i] = new_node;
  }
  for (int i = 0; i < 9; ++i) {
    trees[i] = temp[i];
  }
}

void iter_t_trees(std::array<TTree, 2>& trees) {
  auto smeta = std::make_shared<TNode>();
  smeta->children.push_back(trees[1].root);
  for (const auto& rule : T_RULES) {
    Matrix3d transform = affrot(rule.ang)
  }
}

// function buildSupertiles( sys )
// {
// 	const quad = sys['Delta'].quad;
// 	const R = [-1,0,0,0,1,0];
//
// 	const t_rules = [
// 		[60, 3, 1], [0, 2, 0], [60, 3, 1], [60, 3, 1],
// 		[0, 2, 0], [60, 3, 1], [-120, 3, 3] ];
//
// 	const Ts = [ident];
// 	let total_ang = 0;
// 	let rot = ident;
// 	const tquad = [...quad];
// 	for( const [ang,from,to] of t_rules ) {
// 		total_ang += ang;
// 		if( ang != 0 ) {
// 			rot = trot( radians( total_ang ) );
// 			for( i = 0; i < 4; ++i ) {
// 				tquad[i] = transPt( rot, quad[i] );
// 			}
// 		}
//
// 		const ttt = transTo( tquad[to],
// 			transPt( Ts[Ts.length-1], quad[from] ) );
// 		Ts.push( mul( ttt, rot ) );
// 	}
//
// 	for( let idx = 0; idx < Ts.length; ++idx ) {
// 		Ts[idx] = mul( R, Ts[idx] );
// 	}
//
// 	// Now build the actual supertiles, labelling appropriately.
// 	const super_rules = {
// 		'Gamma' :
// ['Pi','Delta','null','Theta','Sigma','Xi','Phi','Gamma'], 		'Delta'
// :
// ['Xi','Delta','Xi','Phi','Sigma','Pi','Phi','Gamma'], 		'Theta'
// :
// ['Psi','Delta','Pi','Phi','Sigma','Pi','Phi','Gamma'], 		'Lambda'
// :
// ['Psi','Delta','Xi','Phi','Sigma','Pi','Phi','Gamma'], 		'Xi' :
// ['Psi','Delta','Pi','Phi','Sigma','Psi','Phi','Gamma'], 		'Pi' :
// ['Psi','Delta','Xi','Phi','Sigma','Psi','Phi','Gamma'], 		'Sigma'
// :
// ['Xi','Delta','Xi','Phi','Sigma','Pi','Lambda','Gamma'], 		'Phi' :
// ['Psi','Delta','Psi','Phi','Sigma','Pi','Phi','Gamma'], 		'Psi' :
// ['Psi','Delta','Psi','Phi','Sigma','Psi','Phi','Gamma'] }; 	const
// super_quad = [ 		transPt( Ts[6], quad[2] ), 		transPt(
// Ts[5], quad[1]
// ), 		transPt( Ts[3], quad[2] ), 		transPt( Ts[0],
// quad[1] ) ];
//
// }

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

void get_pts(const Node* node, Matrix3d transf, const Tile& shape,
             std::vector<Tile>& shapes) {
  if (node == nullptr) {
    Tile bleh = transf * shape;
    std::cout << transf << '\n';
    std::cout << bleh << '\n';
    shapes.push_back(bleh);
  } else {
    for (const auto& child : node->children) {
      std::cout << "Transformation matrix is: " << transf << '\n';
      std::cout << "Child transform is: " << child.second << '\n';
      get_pts(child.first.get(), transf * child.second, shape, shapes);
    }
  }
}

Matrix3Xd tree_to_tiles(const Tree& tree, const Tile& shape) {
  std::vector<Tile> shapes;
  shapes.reserve(2000);
  get_pts(tree.root.get(), Matrix3d::Identity(), shape, shapes);

  u64 n_cols = 0;
  for (const auto& pts : shapes) {
    n_cols += pts.cols();
  }
  Matrix3Xd all_shapes(3, n_cols);
  u64 cur_col = 0;
  for (const auto& pts : shapes) {
    all_shapes(all, Eigen::seqN(cur_col, pts.cols())) = pts;
    cur_col += pts.cols();
  }
  return all_shapes;
}

s32 to_screen_isotropic(f64 r, f64 start, f64 scale, s32 dim) {
  return (s32)(((r - start) / scale) * (f64)dim);
}

std::array<Color, 14> colors{DARKGRAY,   MAROON,    ORANGE, DARKGREEN, DARKBLUE,
                             DARKPURPLE, DARKBROWN, GRAY,   RED,       GOLD,
                             LIME,       BLUE,      VIOLET, BROWN};

// function buildSupertiles(sys) {
//   const sing = sys['H8'];
//   const comp = sys['H7'];
//
//   const quad = sys['H8'].quad;
//
//   const smeta = new Meta();
//   const rules =
//       [[PI / 3, 2, 0, false], [2 * PI / 3, 2, 0, false], [0, 1, 1, true],
//        [-2 * PI / 3, 2, 2, false], [-PI / 3, 2, 0, false],[0, 2, 0, false]];
//
//   smeta.addChild(sing);
//   for (let r of rules) {
//     if (r[3]) {
//       smeta.addChild(comp.rotateAndMatch(
//           trot(r[0]), r[1],smeta.geoms[smeta.geoms.length - 1].quad[r[2]]));
//     } else {
//       smeta.addChild(sing.rotateAndMatch(
//           trot(r[0]), r[1],smeta.geoms[smeta.geoms.length - 1].quad[r[2]]));
//     }
//   }
//
//   smeta.quad = [
//     smeta.geoms[1].quad[3], smeta.geoms[2].quad[0],smeta.geoms[4].quad[3],
//     smeta.geoms[6].quad[0]
//   ];
//
//   const cmeta = new Meta();
//   cmeta.geoms = smeta.geoms.slice(0, smeta.geoms.length - 1);
//   cmeta.quad = smeta.quad;
//
//   return {'H8' : smeta, 'H7' : cmeta};
// }

int main(int argc, char* argv[]) {
  cxxopts::Options options("makemonotile",
                           "Create a section of the hat monotile tiling, based "

                           "on the code developed by Craig S. Kaplan et al. "
                           "at: https://github.com/isohedral/hatviz");

  options.add_options()("h,help", "Show this help text")(
      "l,level", "Number of metatile iterations.", cxxopts::value<u64>())(
      "a,alength", "length of a-parameter", cxxopts::value<f64>())(
      "p,points", "Draw points."
#ifndef MONOTILE_VISUAL
                  ". Disabled, compile with -DMONOTILE_VISUAL to enable."
#endif // !MONOTILE_VISUAL
      )("t,tiles", "Draw tiles"
#ifndef MONOTILE_VISUAL
                   ". Disabled, compile with -DMONOTILE_VISUAL to enable."
#endif // !MONOTILE_VISUAL
        )("o,output", "Point output.",
          cxxopts::value<std::string>()->default_value("monopts.txt"));
  cxxopts::ParseResult result;

  try {
    result = options.parse(argc, argv);
  } catch (const std::exception& exc) {
    std::cout << options.help() << std::endl;
    return EXIT_FAILURE;
  }

  if (static_cast<bool>(result.count("h"))) {
    std::cout << options.help() << std::endl;
    exit(0);
  }
  const f64 a = result["a"].as<f64>();
  const f64 b = 1 + sqrt3 - a;

  Tile tile = Tile::Zero(3, 14);
  tile(2, 0) = 1;
  for (int i = 0; i < 13; ++i) {
    tile(all, i + 1) = affadd(tile(all, i), EDGES[i].vec(a, b));
  }
  auto keys = std::make_shared<Quad>(3, 4);
  (*keys)(all, 0) = tile(all, 3);
  (*keys)(all, 1) = tile(all, 5);
  (*keys)(all, 2) = tile(all, 7);
  (*keys)(all, 3) = tile(all, 11);
  // Node mystic1{};
  // Node mystic2{};
  // mystic1.quad = std::shared_ptr<Quad>(&keys);
  // mystic2.quad = std::shared_ptr<Quad>(&keys);
  std::array<Tree, 9> categories{};
  for (u32 i = 0; i < 8; ++i) {
    categories[i] = Tree{};
    categories[i].root = std::make_shared<Node>();
    categories[i].root->quad = keys;
  }

  categories[8].root = std::make_shared<Node>();
  categories[8].root->children.push_back({{}, Matrix3d::Identity()});
  categories[8].root->children.push_back(
      {{}, transl2({2 * sqrt3, 6}) * reflect_y() * affrot(M_PI)});
  categories[8].root->quad = keys;

  s32 width = 800;
  s32 height = 800;
  SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_WINDOW_RESIZABLE |
                 FLAG_WINDOW_TRANSPARENT);
  InitWindow(width, height, "raylib test");

  SetTargetFPS(10);
  auto points = tree_to_tiles(categories[8], tile);
  std::cout << "number of points: " << points.cols() << '\n';
  f64 xmin = points(0, all).minCoeff();
  f64 xmax = points(0, all).maxCoeff();
  f64 ymin = points(1, all).minCoeff();
  f64 ymax = points(1, all).maxCoeff();
  f64 exmin = xmin - 0.05 * (xmax - xmin);
  f64 eymin = ymin - 0.05 * (ymax - ymin);
  f64 exmax = xmax + 0.05 * (xmax - xmin);
  f64 eymax = ymax + 0.05 * (ymax - ymin);
  std::cout << "number of tiles " << points.cols() / 14 << '\n';
  f64 max_of_exey = std::max(eymax - eymin, exmax - exmin);
  while (!WindowShouldClose()) {
    width = GetScreenWidth();
    height = GetScreenHeight();
    s32 min_of_wh = std::min(width, height);

    BeginDrawing();
    ClearBackground(WHITE);
    if (result["t"].as<bool>()) {
      for (s32 i = 0; i < points.cols() / 14; ++i) {
        for (s32 j = 0; j < 14; ++j) {
          // std::cout << "Attempting to draw point: {" << points(0, i * 14 + j)
          //           << ", " << points(0, i * 14 + j) << "}\n";
          DrawLineEx(
              {static_cast<f32>(to_screen_isotropic(
                   points(0, i * 14 + j), exmin, max_of_exey, min_of_wh)),
               static_cast<f32>(to_screen_isotropic(
                   points(1, i * 14 + j), eymin, max_of_exey, min_of_wh))},
              {static_cast<f32>(
                   to_screen_isotropic(points(0, i * 14 + (j + 1) % 14), exmin,
                                       max_of_exey, min_of_wh)),
               static_cast<f32>(
                   to_screen_isotropic(points(1, i * 14 + (j + 1) % 14), eymin,
                                       max_of_exey, min_of_wh))},
              4.0, colors[i]);
        }
      }
    }
    // if (result["p"].as<bool>()) {
    //   for (int i = 0; i < unique_pts.cols(); ++i) {
    //     DrawCircle(to_screen_isotropic(unique_pts(0, i), exmin, max_of_exey,
    //                                    min_of_wh),
    //                to_screen_isotropic(unique_pts(1, i), eymin, max_of_exey,
    //                                    min_of_wh),
    //                4, RED);
    //   }
    // }
    EndDrawing();
  }
  CloseWindow();
}
