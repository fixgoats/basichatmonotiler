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

typedef std::array<std::shared_ptr<Vector3d>, 4>
    Quad; // Eigen::Matrix<f64, 3, 4> Quad;
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

enum class Len : bool {
  a,
  b,
};

template <class T, size_t Cap>
struct SmallArr : std::array<T, Cap> {
  size_t len;

  struct Iterator {
    T* m_ptr;

    Iterator& operator++() {
      this->m_ptr++;
      return *this;
    }
    Iterator& operator--() {
      this->m_ptr--;
      return *this;
    }
    Iterator operator++(int) {
      Iterator tmp = *this;
      this->m_ptr++;
      return tmp;
    }
    Iterator operator--(int) {
      Iterator tmp = *this;
      this->m_ptr--;
      return tmp;
    }
    T& operator*() { return *this->m_ptr; }
    bool operator==(const Iterator& other) const {
      return this->m_ptr == other.m_ptr;
    }
    bool operator!=(const Iterator& other) const {
      return this->m_ptr != other.m_ptr;
    }
  };
  struct ConstIterator {
    const T* m_ptr;

    ConstIterator& operator++() {
      this->m_ptr++;
      return *this;
    }
    ConstIterator& operator--() {
      this->m_ptr--;
      return *this;
    }
    ConstIterator operator++(int) {
      Iterator tmp = *this;
      this->m_ptr++;
      return tmp;
    }
    ConstIterator operator--(int) {
      Iterator tmp = *this;
      this->m_ptr--;
      return tmp;
    }
    const T& operator*() { return *this->m_ptr; }
    bool operator==(const ConstIterator& other) const {
      return this->m_ptr == other.m_ptr;
    }
    bool operator!=(const ConstIterator& other) const {
      return this->m_ptr != other.m_ptr;
    }
  };

  constexpr SmallArr() = default;

  template <class... Args>
  constexpr SmallArr(Args&&... args)
    requires(std::is_same_v<std::common_type_t<Args...>, T>)
      : std::array<T, Cap>{std::forward<Args>(args)...}, len{sizeof...(Args)} {}
  constexpr SmallArr(size_t s) : len{s}, std::array<T, Cap>{} {}

  [[nodiscard]] constexpr T operator[](auto i) const {
    ASSERT(i < len, "Attempted out of bounds access.");
    return this->data()[i];
  }
  [[nodiscard]] constexpr const T& operator[](auto i) const {
    ASSERT(i < len, "Attempted out of bounds access.");
    return this->data()[i];
  }

  [[nodiscard]] T& operator[](auto i) {
    ASSERT(i < len, "Attempted out of bounds access.");
    return this->data()[i];
  }

  [[nodiscard]] constexpr T front() const {
    ASSERT(len > 0, "Attempted to access empty array.");
    return this->data()[0];
  }

  [[nodiscard]] T& front() {
    ASSERT(len > 0, "Attempted to access empty array.");
    return this->data()[0];
  }

  [[nodiscard]] constexpr T back() const {
    ASSERT(len > 0, "Attempted to access empty array.");
    return this->data()[len - 1];
  }

  [[nodiscard]] T& back() {
    ASSERT(len > 0, "Attempted to access empty array.");
    return this->data()[len - 1];
  }

  constexpr void push_back(T x) {
    ASSERT(len < Cap, "Pushing back would exceed capacity.");
    this->data()[len] = x;
    len += 1;
  }

  template <class... Args>
  constexpr void emplace_back(Args&&... args) {
    ASSERT(len < Cap, "Emplacing back would exceed capacity.");
    this->data()[len] = T{std::forward<Args>(args)...};
    len += 1;
  }

  [[nodiscard]] ConstIterator cbegin() const {
    return ConstIterator{this->data()};
  }
  [[nodiscard]] ConstIterator cend() const {
    return ConstIterator{this->data() + this->len};
  }
  [[nodiscard]] ConstIterator begin() const { return cbegin(); }
  [[nodiscard]] ConstIterator end() const { return cend(); }
  Iterator begin() { return Iterator{this->data()}; }
  Iterator end() { return Iterator{this->data() + this->len}; }
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

Quad quad_map(const Quad& q, Matrix3d t) {
  Quad ret;
  std::transform(q.cbegin(), q.cend(), ret.begin(),
                 [t](std::shared_ptr<Vector3d> v) {
                   return std::make_shared<Vector3d>(t * (*v));
                 });
  return ret;
}

void quad_transform(Quad& q, Matrix3d t) {
  for (auto& v : q) {
    *v = t * (*v);
  }
}

struct TNode {
  SmallArr<std::shared_ptr<TNode>, 7> children;
  // Matrix3d transform;
  Quad quad;
  std::optional<Tile> shape;

  ~TNode() {
    std::cout << "~TNode called\n";
    std::cout << "Had " << children.len << " children\n";
  }
  void get_pts(std::vector<Vector3d>& pts, u32 iter_depth = 0) {
    if (shape.has_value()) {
      std::cout << "Iteration depth: " << iter_depth << std::endl;
      for (const auto& col : shape.value().colwise()) {
        pts.emplace_back(col);
      }
    }
    for (const auto& child : children) {
      child->get_pts(pts, iter_depth + 1);
    }
  }

  void translate_in_place(Vector3d dp) {
    if (shape.has_value()) {
      shape = transl3(dp) * shape.value();
    }
    for (const auto& child : children) {
      child->translate_in_place(dp);
    }
    for (auto& v : quad) {
      *v = transl3(dp) * *v;
    };
  }

  friend std::ostream& operator<<(std::ostream& os, const TNode& q);

  std::shared_ptr<TNode> rotate_and_match(Matrix3d t, u32 i = 0) {
    auto ret = std::make_shared<TNode>();
    std::cout << "rotate_and_match, " << i << "th/st/nd/rd level";
    for (u32 i = 0; i < children.len; i++) {
      ret->children.push_back(children[i]->rotate_and_match(t, i + 1));
    }
    if (shape.has_value()) {
      ret->shape = t * shape.value();
      std::cout << "made new tile\n";
    }
    ret->quad = quad_map(quad, t);
    // std::cout << "q1: " << *ret->quad[0] << '\n';
    // std::cout << "q2: " << *ret->quad[1] << '\n';
    // std::cout << "q3: " << *ret->quad[2] << '\n';
    // std::cout << "q4: " << *ret->quad[3] << '\n';
    return ret;
  }

  std::shared_ptr<TNode> rotate_and_match(Matrix3d t, u32 j, Vector3d P,
                                          u32 i = 0) {
    auto ret = std::make_shared<TNode>();
    std::cout << "rotate_and_match, " << i << "th/st/nd/rd level";
    // ret->transform = t * transform;

    for (u32 i = 0; i < children.len; i++) {
      ret->children.push_back(children[i]->rotate_and_match(t, i + 1));
    }
    if (shape.has_value()) {
      ret->shape = t * shape.value();
      std::cout << "made new tile\n";
    }
    ret->quad = quad_map(quad, t);
    ret->translate_in_place(affsub(P, *(ret->quad[j])));
    // std::cout << "q1: " << *ret->quad[0] << '\n';
    // std::cout << "q2: " << *ret->quad[1] << '\n';
    // std::cout << "q3: " << *ret->quad[2] << '\n';
    // std::cout << "q4: " << *ret->quad[3] << '\n';
    return ret;
  }
};

Matrix3Xd quad_to_mat(Quad q) {
  Matrix3Xd ret(3, 4);
  for (s32 i = 0; i < 4; i++) {
    ret(all, i) = *q[i];
  }
  return ret;
}

std::ostream& operator<<(std::ostream& os, const TNode& n) {
  os << "{\n";
  os << "  quad: " << quad_to_mat(n.quad) << "\n";
  if (n.shape.has_value()) {
    os << "  shape: " << n.shape.value() << "\n";
  }
  for (const auto& child : n.children) {
    os << "  child" << child << ": " << *child << "\n";
  }
  os << "}\n";
}

struct TTree {
  std::shared_ptr<TNode> root;

  Matrix3Xd get_pts() {
    std::vector<Vector3d> pts;
    pts.reserve(50000);
    root->get_pts(pts);
    Matrix3Xd points(3, pts.size());
    for (u32 i = 0; i < pts.size(); i++) {
      points(all, i) = pts[i];
    }
    return points;
  }
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

// constexpr std::array<TRule, 6> T_RULES = {{
//     {.ang = 0, .i = 2, .j = 0, .singcomp = false},
//     {.ang = 0, .i = 2, .j = 0, .singcomp = false},
//     {.ang = 0, .i = 1, .j = 1, .singcomp = true},
//     {.ang = 0, .i = 2, .j = 2, .singcomp = false},
//     {.ang = 0, .i = 2, .j = 0, .singcomp = false},
//     {.ang = 0, .i = 2, .j = 0, .singcomp = false},
// }};

constexpr std::array<TRule, 6> T_RULES = {{
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

// void iter_trees(std::array<Tree, 9> trees) {
//   const Quad ref = trees[static_cast<u8>(Label::Delta)].root->quad;
//   f64 total_ang = 0;
//   Matrix3d rot = Matrix3d::Identity();
//
//   Quad tquad{};
//   std::array<Matrix3d, 8> transforms{};
//   transforms[0] = Matrix3d::Identity();
//   for (u32 i = 0; i < 7; ++i) {
//     total_ang += T_RULES[i].ang;
//     if (T_RULES[i].ang != 0) {
//       rot = affrot(total_ang);
//       tquad = rot * (*ref);
//     }
//     const Vector3d ttt = affsub(transforms[i] * (*ref)(all, T_RULES[i].i),
//                                 tquad(all, T_RULES[i].j));
//     transforms[i + 1] = translate_by3(rot, ttt);
//   }
//   for (auto& transform : transforms) {
//     transform = reflect_y(transform);
//   }
//
//   auto super_quad = std::make_shared<Quad>();
//   (*super_quad)(all, 0) = transforms[6] * (*ref)(all, 2);
//   (*super_quad)(all, 1) = transforms[5] * (*ref)(all, 1);
//   (*super_quad)(all, 2) = transforms[3] * (*ref)(all, 2);
//   (*super_quad)(all, 3) = transforms[0] * (*ref)(all, 2);
//   // transPt( Ts[6], quad[2] ),
//   // transPt( Ts[5], quad[1] ),
//   // transPt( Ts[3], quad[2] ),
//   // transPt( Ts[0], quad[1] ) ];
//   std::array<std::shared_ptr<Node>, 9> temp{};
//   for (u32 i = 0; i < 9; ++i) {
//     auto new_node = std::make_shared<Node>();
//     for (u32 j = 0; j < SUPER_RULES[i].size; ++j) {
//       new_node->children.push_back(
//           {trees[static_cast<u8>(SUPER_RULES[i][j])].root, transforms[j]});
//     }
//     new_node->quad = super_quad;
//     temp[i] = new_node;
//   }
//   for (int i = 0; i < 9; ++i) {
//     trees[i] = temp[i];
//   }
// }

void iter_t_trees(std::array<TTree, 2>& trees) {
  auto smeta = std::make_shared<TNode>();
  smeta->children.emplace_back(trees[0].root);
  for (const auto& rule : T_RULES) {
    Matrix3d transform = affrot(rule.ang);
    if (rule.singcomp) {
      smeta->children.emplace_back(trees[1].root->rotate_and_match(
          affrot(rule.ang), rule.i, *(smeta->children.back()->quad[rule.j])));
    } else {
      smeta->children.emplace_back(trees[0].root->rotate_and_match(
          affrot(rule.ang), rule.i, *(smeta->children.back()->quad[rule.j])));
    }
  }
  smeta->quad = {smeta->children[1]->quad[3], smeta->children[2]->quad[0],
                 smeta->children[4]->quad[3], smeta->children[6]->quad[0]};
  for (const auto& v : smeta->quad) {
    std::cout << "smeta p: " << *v << "\n";
  }

  std::shared_ptr<TNode> cmeta = std::make_shared<TNode>();

  for (u32 i = 0; i < smeta->children.len - 1; ++i) {
    cmeta->children.emplace_back(smeta->children[i]);
  }

  cmeta->quad = smeta->quad;
  for (const auto& v : smeta->quad) {
    std::cout << "cmeta p: " << *v << "\n";
  }
  trees[0].root = smeta;
  trees[1].root = cmeta;
}
// rotateAndMatch(T, qidx, P) {
//   const ret = new Meta();
//   ret.geoms = this.geoms.map(g = > g.rotateAndMatch(T, -1));
//   ret.quad = this.quad.map(p = > transAB(T, p));
//   if (qidx >= 0) {
//     ret.translateInPlace(psub(P, ret.quad[qidx]));
//   }
//   return ret;
// }

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

// void get_pts(const TNode* node, Matrix3d transf, std::vector<Tile>& shapes) {
//   if (node->children.size == 0) {
//     Tile bleh = transf * node->shape.value();
//     std::cout << transf << '\n';
//     std::cout << bleh << '\n';
//     shapes.push_back(bleh);
//   } else {
//     for (const auto& child : node->children) {
//       std::cout << "Transformation matrix is: " << transf << '\n';
//       // std::cout << "Child transform is: " << child.second << '\n';
//       get_pts(child.get(), shapes);
//     }
//   }
// }

// Matrix3Xd tree_to_tiles(const TTree& tree) {
//   std::vector<Tile> shapes;
//   shapes.reserve(2000);
//   get_pts(tree.root.get(), Matrix3d::Identity(), shapes);
//
//   u64 n_cols = 0;
//   for (const auto& pts : shapes) {
//     n_cols += pts.cols();
//   }
//   Matrix3Xd all_shapes(3, n_cols);
//   u64 cur_col = 0;
//   for (const auto& pts : shapes) {
//     all_shapes(all, Eigen::seqN(cur_col, pts.cols())) = pts;
//     cur_col += pts.cols();
//   }
//   return all_shapes;
// }

s32 to_screen_isotropic(f64 r, f64 start, f64 scale, s32 dim) {
  return (s32)(((r - start) / scale) * (f64)dim);
}

#ifdef MONOTILE_VISUAL
std::array<Color, 14> colors{DARKGRAY,   MAROON,    ORANGE, DARKGREEN, DARKBLUE,
                             DARKPURPLE, DARKBROWN, GRAY,   RED,       GOLD,
                             LIME,       BLUE,      VIOLET, BROWN};
#endif

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

  Tile tile1 = Tile::Zero(3, 14);
  tile1(2, all) = Eigen::VectorXd::Ones(14);

  std::cout << tile1(all, 0) << std::endl;
  for (int i = 0; i < 13; ++i) {
    tile1(all, i + 1) = affadd(tile1(all, i), EDGES[i].vec(a, b));
    std::cout << tile1(all, i + 1) << std::endl;
  }
  Tile tile2 = reflect_x() * tile1;

  Quad keys{};
  keys[0] = std::make_shared<Vector3d>(tile1(all, 1));
  keys[1] = std::make_shared<Vector3d>(tile1(all, 3));
  keys[2] = std::make_shared<Vector3d>(tile1(all, 9));
  keys[3] = std::make_shared<Vector3d>(tile1(all, 13));
  std::array<TTree, 2> categories{};
  for (const auto& k : keys) {
    std::cout << "quad p: " << *k << std::endl;
  }

  categories[0].root = std::make_shared<TNode>(TNode({}, keys, tile1));
  ; // std::make_shared<TNode>(TNode{{node1}, keys, {}});
  tile2 = transl3(affsub(tile1(all, 11), tile2(all, 5))) * tile2;
  std::cout << "Tile 1:\n";
  std::cout << tile1 << '\n';
  std::cout << "Tile 2:\n";
  std::cout << tile2 << '\n';
  categories[1].root = std::make_shared<TNode>(
      TNode{{std::make_shared<TNode>(TNode({}, keys, tile1)),
             std::make_shared<TNode>(TNode({{}, keys, tile2}))},
            keys,
            {}});
  // for (u32 i = 0; i < 2; ++i) {
  //   categories[i] = Tree{};
  //   categories[i].root = std::make_shared<Node>();
  //   categories[i].root->quad = keys;
  // }

  // categories[8].root = std::make_shared<Node>();
  // categories[8].root->children.push_back({{}, Matrix3d::Identity()});
  // categories[8].root->children.push_back(
  //     {{}, transl2({2 * sqrt3, 6}) * reflect_y() * affrot(M_PI)});
  // categories[8].root->quad = keys;

  for (u64 i = 0; i < result["l"].as<u64>(); ++i) {
    iter_t_trees(categories);
    std::cout << "bleh\n";
  }
  s32 width = 800;
  s32 height = 800;
  SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_WINDOW_RESIZABLE |
                 FLAG_WINDOW_TRANSPARENT);
  InitWindow(width, height, "raylib test");

  SetTargetFPS(10);
  auto points = categories[0].get_pts();
  f64 xmin = points(0, all).minCoeff();
  f64 xmax = points(0, all).maxCoeff();
  f64 ymin = points(1, all).minCoeff();
  f64 ymax = points(1, all).maxCoeff();
  f64 exmin = xmin - 0.05 * (xmax - xmin);
  f64 eymin = ymin - 0.05 * (ymax - ymin);
  f64 exmax = xmax + 0.05 * (xmax - xmin);
  f64 eymax = ymax + 0.05 * (ymax - ymin);
  f64 max_of_exey = std::max(eymax - eymin, exmax - exmin);
  u32 cat_index = 0;
  while (!WindowShouldClose()) {
    width = GetScreenWidth();
    height = GetScreenHeight();
    s32 min_of_wh = std::min(width, height);

    if (IsKeyPressed(KEY_ONE)) {
      cat_index = 0;
      points = categories[0].get_pts();
    }
    if (IsKeyPressed(KEY_TWO)) {
      cat_index = 1;
      points = categories[1].get_pts();
    }
    if (IsKeyPressed(KEY_N)) {
      iter_t_trees(categories);
      points = categories[cat_index].get_pts();
    }
    f64 xmin = points(0, all).minCoeff();
    f64 xmax = points(0, all).maxCoeff();
    f64 ymin = points(1, all).minCoeff();
    f64 ymax = points(1, all).maxCoeff();
    f64 exmin = xmin - 0.05 * (xmax - xmin);
    f64 eymin = ymin - 0.05 * (ymax - ymin);
    f64 exmax = xmax + 0.05 * (xmax - xmin);
    f64 eymax = ymax + 0.05 * (ymax - ymin);
    f64 max_of_exey = std::max(eymax - eymin, exmax - exmin);
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
              4.0, colors[i % 14]);
        }
      }
    }
    if (result["p"].as<bool>()) {
      for (int i = 0; i < unique_pts.cols(); ++i) {
        DrawCircle(to_screen_isotropic(unique_pts(0, i), exmin, max_of_exey,
                                       min_of_wh),
                   to_screen_isotropic(unique_pts(1, i), eymin, max_of_exey,
                                       min_of_wh),
                   4, RED);
      }
    }
    EndDrawing();
  }
  CloseWindow();
}
