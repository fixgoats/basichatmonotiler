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
typedef std::variant<Quad, Tile> shape_var_t;

enum class Len : bool {
  a,
  b,
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
  Null = 9,
  Mys1 = 10,
  Mys2 = 11,
};

constexpr std::array<Label, 9> LABELS{
    Label::Delta, Label::Theta, Label::Lambda, Label::Xi,    Label::Pi,
    Label::Sigma, Label::Phi,   Label::Psi,    Label::Gamma,
};
constexpr std::array<std::array<Label, 8>, 9> SUPER_RULES = {
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
     {Label::Pi, Label::Delta, Label::Null, Label::Theta, Label::Sigma,
      Label::Xi, Label::Phi, Label::Gamma}}};

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
  std::vector<std::shared_ptr<Node>> children;
  Matrix3d transform;
  Quad quad;
  Label lab;

  Node() = default;
  Node(Matrix3d tr, Quad q, Label label) : transform{tr}, quad{q}, lab{label} {}
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

struct Rule {
  std::array<f64, 3> num;
  bool huh;
};

struct TRule {
  f64 ang;
  u32 i;
  u32 j;
};

constexpr std::array<TRule, 7> T_RULES = {{
    {.ang = pi / 3, .i = 3, .j = 1},
    {.ang = 0, .i = 2, .j = 0},
    {.ang = pi / 3, .i = 3, .j = 1},
    {.ang = pi / 3, .i = 3, .j = 1},
    {.ang = 0, .i = 2, .j = 0},
    {.ang = pi / 3, .i = 3, .j = 1},
    {.ang = -2 * pi / 3, .i = 3, .j = 3},
}};

struct Pt2 : std::array<f64, 2> {
  static constexpr size_t DIM = 2;
};

constexpr f64 hsq3 = 0.5 * std::numbers::sqrt3;

static const std::array<Vector3d, 12> dirs{{
    {1, 0, 1},
    {hsq3, 0.5, 1},
    {0.5, hsq3, 1},
    {0, 1, 1},
    {-0.5, -hsq3, 1},
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
      return scale(dirs[dir], a);
      break;
    }
    case Len::b: {
      return scale(dirs[dir], b);
      break;
    }
    }
  }
};

constexpr std::array<Edge, 13> edges{{
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
  const Quad ref = trees[static_cast<u8>(Label::Delta)].root->quad;
  f64 total_ang = 0;
  Matrix3d rot = Matrix3d::Identity();

  std::array<Matrix3d, 8> transforms{};
  transforms[0] = Matrix3d::Identity();
  for (u32 i = 0; i < 7; ++i) {
    total_ang += T_RULES[i].ang;
    if (T_RULES[i].ang != 0) {
      rot = affrot(total_ang);
    }
    const Vector3d ttt =
        affsub(transforms[i] * ref(all, T_RULES[i].i), ref(all, T_RULES[i].j));
    transforms[i + 1] = translate_by3(rot, ttt);
    for (auto& transform : transforms) {
      transform = reflect_y(transform);
    }
  }
  // 	const ret = {};
  //
  // 	for( const [lab, subs] of Object.entries( super_rules ) ) {
  // 		const sup = new Meta();
  // 		for( let idx = 0; idx < 8; ++idx ) {
  // 			if( subs[idx] == 'null' ) {
  // 				continue;
  // 			}
  // 			sup.addChild( sys[subs[idx]], Ts[idx] );
  // 		}
  // 		sup.quad = super_quad;
  //
  // 		ret[lab] = sup;
  // 	}
  //
  // 	return ret;
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
int main(int argc, char* argv[]) {
  cxxopts::Options options("makemonotile",
                           "Create a section of the hat monotile tiling, based "
                           "on the code developed by Craig S. Kaplan et al. "
                           "at: https://github.com/isohedral/hatviz");

  options.add_options()("h,help", "Show this help text")(
      "l,level", "Number of metatile iterations.", cxxopts::value<u64>())(
      "a", "length of a-parameter", cxxopts::value<f64>())(
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

  if (static_cast<bool>(result["h"].count())) {
    std::cout << options.help() << std::endl;
    exit(0);
  }
  const f64 a = result["a"].as<f64>();
  const f64 b = sqrt3 - a;

  Eigen::Matrix<f64, 3, 14> tile = Eigen::Matrix<f64, 3, 14>::Zero(3, 14);
  for (int i = 0; i < 13; ++i) {
    tile(all, i + 1) = affadd(tile(all, i), edges[i].vec(a, b));
  }
  Quad keys(3, 4);
  keys(all, 0) = tile(all, 3);
  keys(all, 1) = tile(all, 5);
  keys(all, 2) = tile(all, 7);
  keys(all, 3) = tile(all, 11);
  Node mystic1(Matrix3d::Identity(), {}, Label::Mys1);
  Node mystic2(translate_by3(affrot(M_PI / 6), tile(all, 8)), {}, Label::Mys2);
  std::array<Tree, 9> categories{};
  for (u32 i = 0; i < 8; ++i) {
    categories[i] =
        Tree(std::make_shared<Node>(Node(Matrix3d::Identity(),
                                          keys,
                                          static_cast<Label>(i)));
  }
  Node gamma{.children = {std::shared_ptr<Node>{&mystic1},
                          std::shared_ptr<Node>{&mystic2}},
             .transform = Matrix3d::Identity(),
             .quad = keys,
             .lab = Label::Gamma};
  categories[8] = Tree{.root = std::shared_ptr<Node>(&gamma)};
}

// function buildSpectreBase( curved )
// {
// 	const spectre = [
// 		pt(0, 0),
// 		pt(1.0, 0.0),
// 		pt(1.5, -0.8660254037844386),
// 		pt(2.366025403784439, -0.36602540378443865),
// 		pt(2.366025403784439, 0.6339745962155614),
// 		pt(3.366025403784439, 0.6339745962155614),
// 		pt(3.866025403784439, 1.5),
// 		pt(3.0, 2.0),
// 		pt(2.133974596215561, 1.5),
// 		pt(1.6339745962155614, 2.3660254037844393),
// 		pt(0.6339745962155614, 2.3660254037844393),
// 		pt(-0.3660254037844386, 2.3660254037844393),
// 		pt(-0.866025403784439, 1.5),
// 		pt(0.0, 1.0)
// 	];
//
// 	const spectre_keys = [
// 		spectre[3], spectre[5], spectre[7], spectre[11]
// 	];
//
// 	const ret = {};
//
// 	for( lab of ['Delta', 'Theta', 'Lambda', 'Xi',
// 				 'Pi', 'Sigma', 'Phi', 'Psi'] ) {
// 		if( curved ) {
// 			ret[lab] = new CurvyShape( spectre, spectre_keys, lab );
// 		} else {
// 			ret[lab] = new Shape( spectre, spectre_keys, lab );
// 		}
// 	}
//
// 	const mystic = new Meta();
// 	if( curved ) {
// 		mystic.addChild(
// 			new CurvyShape( spectre, spectre_keys, 'Gamma1' ), ident
// ); 		mystic.addChild(			new CurvyShape(
// spectre, spectre_keys, 'Gamma2' ), 				mul(
// ttrans( spectre[8].x, spectre[8].y ), trot( PI / 6 ) ) ); 	} else {
// 		mystic.addChild( new Shape( spectre, spectre_keys, 'Gamma1' ),
// ident ); 		mystic.addChild( new Shape( spectre, spectre_keys,
// 'Gamma2' ), 			mul( ttrans( spectre[8].x, spectre[8].y
// ), trot( PI / 6 ) ) );
// 	}
// 	mystic.quad = spectre_keys;
// 	ret['Gamma'] = mystic;
//
// 	return ret;
// }
// function buildBaseTiles()
// {
//   // Schematic description of the edges of a shape in the hat
//   // continuum.  Each edge's length is one of 'a' or 'b', and the
//   // direction d gives the orientation of d*30 degrees relative to
//   // the positive X axis.
//   const edges = [
//           ['a',0], ['a',2], ['b',11], ['b',1], ['a',4], ['a',2],
//           ['b',5], ['b',3], ['a',6], ['a',8], ['a',8], ['a',10], ['b',7] ];
//   const hr3 = 0.5*1.7320508075688772;
//   const dirs = [pt(1,0), pt(hr3,0.5), pt(0.5,hr3),
//           pt(0,1), pt(-0.5,hr3), pt(-hr3,0.5),
//           pt(-1,0), pt(-hr3,-0.5), pt(-0.5,-hr3),
//           pt(0,-1), pt(0.5,-hr3), pt(hr3,-0.5)];
//
//   let prev = pt(makeAB(0,0),makeAB(0,0));
//   const pts = [prev];
//
//   for( let e of edges ) {
//     if( e[0] == 'a' ) {
//       prev = pt(
//               addAB( prev.x, makeAB( dirs[e[1]].x, 0 ) ),
//               addAB( prev.y, makeAB( dirs[e[1]].y, 0 ) ) );
//     } else {
//       prev = pt(
//               addAB( prev.x, makeAB( 0, dirs[e[1]].x ) ),
//               addAB( prev.y, makeAB( 0, dirs[e[1]].y ) ) );
//     }
//
//     pts.push( prev );
//   }
//
//   const quad = [pts[1], pts[3], pts[9], pts[13]];
//   const ret = {};
//
//   ret['H8'] = new Shape( pts, quad, 'single' );
//
//   const fpts = [];
//   const len = pts.length;
//   for( let idx = 0; idx < len; ++idx ) {
//     const p = pts[len-1-idx];
//     fpts.push( pt( p.x, scaleAB( p.y, -1 ) ) );
//   }
//   const dp = psub( pts[0], fpts[5] );
//   for( let idx = 0; idx < len; ++idx ) {
//     fpts[idx] = padd( fpts[idx], dp );
//   }
//
//   const comp = new Meta();
//   comp.addChild( new Shape( pts, quad, 'unflipped' ) );
//   comp.addChild( new Shape( fpts, quad , 'flipped' ) );
//   comp.quad = quad;
//   ret['H7'] = comp;
//
//   return ret;
// }
//
//
//
// function buildSupertiles( sys )
// {
//   const sing = sys['H8'];
//   const comp = sys['H7'];
//
//   const quad = sys['H8'].quad;
//
//   const smeta = new Meta();
//   const rules = [
//           [PI/3, 2, 0, false],
//           [2*PI/3, 2, 0, false],
//           [0, 1, 1, true],
//           [-2*PI/3, 2, 2, false],
//           [-PI/3, 2, 0, false],
//           [0, 2, 0, false] ];
//
//   smeta.addChild( sing );
//   for( let r of rules ) {
//     if( r[3] ) {
//       smeta.addChild( comp.rotateAndMatch( trot( r[0] ), r[1],
//               smeta.geoms[smeta.geoms.length-1].quad[r[2]] ) );
//     } else {
//       smeta.addChild( sing.rotateAndMatch( trot( r[0] ), r[1],
//               smeta.geoms[smeta.geoms.length-1].quad[r[2]] ) );
//     }
//   }
//
//   smeta.quad = [
//           smeta.geoms[1].quad[3], smeta.geoms[2].quad[0],
//           smeta.geoms[4].quad[3], smeta.geoms[6].quad[0] ];
//
//   const cmeta = new Meta();
//   cmeta.geoms = smeta.geoms.slice( 0, smeta.geoms.length - 1 );
//   cmeta.quad = smeta.quad;
//
//   return { 'H8' : smeta, 'H7' : cmeta };
// }
//
// Lýst betur á þessa útgáfu.
// const ident = [1,0,0,0,1,0];
//
// let to_screen = [20, 0, 0, 0, -20, 0];
// let lw_scale = 1;
//
// let sys;
//
// let scale_centre;
// let scale_start;
// let scale_ts;
//
// let reset_but;
// let tile_sel;
// let shape_sel;
// let colscheme_sel;
//
// let subst_button;
// let translate_button;
// let scale_button;
// let dragging = false;
// let uibox = true;
//
// const tile_names = [
// 	'Gamma', 'Delta', 'Theta', 'Lambda', 'Xi',
// 	'Pi', 'Sigma', 'Phi', 'Psi' ];
//
// // Color map from Figure 5.3
// const colmap53 = {
// 	'Gamma' : [203, 157, 126],
// 	'Gamma1' : [203, 157, 126],
// 	'Gamma2' : [203, 157, 126],
// 	'Delta' : [163, 150, 133],
// 	'Theta' : [208, 215, 150],
// 	'Lambda' : [184, 205, 178],
// 	'Xi' : [211, 177, 144],
// 	'Pi' : [218, 197, 161],
// 	'Sigma' : [191, 146, 126],
// 	'Phi' : [228, 213, 167],
// 	'Psi' : [224, 223, 156] };
//
// const colmap_orig = {
// 	'Gamma' : [255, 255, 255],
// 	'Gamma1' : [255, 255, 255],
// 	'Gamma2' : [255, 255, 255],
// 	'Delta' : [220, 220, 220],
// 	'Theta' : [255, 191, 191],
// 	'Lambda' : [255, 160, 122],
// 	'Xi' : [255, 242, 0],
// 	'Pi' : [135, 206, 250],
// 	'Sigma' : [245, 245, 220],
// 	'Phi' : [0, 255, 0],
// 	'Psi' : [0, 255, 255] };
//
// const colmap_mystics = {
// 	'Gamma' : [196, 201, 169],
// 	'Gamma1' : [196, 201, 169],
// 	'Gamma2' : [156, 160, 116],
// 	'Delta' : [247, 252, 248],
// 	'Theta' : [247, 252, 248],
// 	'Lambda' : [247, 252, 248],
// 	'Xi' : [247, 252, 248],
// 	'Pi' : [247, 252, 248],
// 	'Sigma' : [247, 252, 248],
// 	'Phi' : [247, 252, 248],
// 	'Psi' : [247, 252, 248] };
//
// const colmap_pride = {
// 	'Gamma' : [255, 255, 255],
// 	'Gamma1' : [97, 57, 21],
// 	'Gamma2' : [0, 0, 0],
// 	'Delta' : [2, 129, 33],
// 	'Theta' : [0, 76, 255],
// 	'Lambda' : [118, 0, 136],
// 	'Xi' : [229, 0, 0],
// 	'Pi' : [255, 175, 199],
// 	'Sigma' : [115, 215, 238],
// 	'Phi' : [255, 141, 0],
// 	'Psi' : [255, 238, 0] };
//
// let colmap = colmap_pride;
//
// function pt( x, y )
// {
// 	return { x : x, y : y };
// }
//
// // Affine matrix inverse
// function inv( T ) {
// 	const det = T[0]*T[4] - T[1]*T[3];
// 	return [T[4]/det, -T[1]/det, (T[1]*T[5]-T[2]*T[4])/det,
// 		-T[3]/det, T[0]/det, (T[2]*T[3]-T[0]*T[5])/det];
// };
//
// // Affine matrix multiply
// function mul( A, B )
// {
// 	return [A[0]*B[0] + A[1]*B[3],
// 		A[0]*B[1] + A[1]*B[4],
// 		A[0]*B[2] + A[1]*B[5] + A[2],
//
// 		A[3]*B[0] + A[4]*B[3],
// 		A[3]*B[1] + A[4]*B[4],
// 		A[3]*B[2] + A[4]*B[5] + A[5]];
// }
//
// function padd( p, q )
// {
// 	return { x : p.x + q.x, y : p.y + q.y };
// }
//
// function psub( p, q )
// {
// 	return { x : p.x - q.x, y : p.y - q.y };
// }
//
// function pframe( o, p, q, a, b )
// {
// 	return { x : o.x + a*p.x + b*q.x, y : o.y + a*p.y + b*q.y };
// }
//
// // Rotation matrix
// function trot( ang )
// {
// 	const c = cos( ang );
// 	const s = sin( ang );
// 	return [c, -s, 0, s, c, 0];
// }
//
// // Translation matrix
// function ttrans( tx, ty )
// {
// 	return [1, 0, tx, 0, 1, ty];
// }
//
// function transTo( p, q )
// {
// 	return ttrans( q.x - p.x, q.y - p.y );
// }
//
// function rotAbout( p, ang )
// {
// 	return mul( ttrans( p.x, p.y ),
// 		mul( trot( ang ), ttrans( -p.x, -p.y ) ) );
// }
//
// // Matrix * point
// function transPt( M, P )
// {
// 	return pt(M[0]*P.x + M[1]*P.y + M[2], M[3]*P.x + M[4]*P.y + M[5]);
// }
//
// // Match unit interval to line segment p->q
// function matchSeg( p, q )
// {
// 	return [q.x-p.x, p.y-q.y, p.x,  q.y-p.y, q.x-p.x, p.y];
// };
//
// // Match line segment p1->q1 to line segment p2->q2
// function matchTwo( p1, q1, p2, q2 )
// {
// 	return mul( matchSeg( p2, q2 ), inv( matchSeg( p1, q1 ) ) );
// };
//
// function drawPolygon( shape, T, f, s, w )
// {
// 	if( f != null ) {
// 		fill( ...f );
// 	} else {
// 		noFill();
// 	}
// 	if( s != null ) {
// 		stroke( 0 );
// 		strokeWeight( w ) ; // / lw_scale );
// 	} else {
// 		noStroke();
// 	}
// 	beginShape();
// 	for( let p of shape ) {
// 		const tp = transPt( T, p );
// 		vertex( tp.x, tp.y );
// 	}
// 	endShape( CLOSE );
// }
//
// function streamPolygon( shape, T, f, s, w )
// {
//
// }
//
// class Shape
// {
// 	constructor( pts, quad, label )
// 	{
// 		this.pts = pts;
// 		this.quad = quad;
// 		this.label = label;
// 	}
//
// 	draw( S )
// 	{
// 		drawPolygon( this.pts, S, colmap[this.label], [0,0,0], 0.1 );
// 	}
//
// 	streamSVG( S, stream )
// 	{
// 		var s = '<polygon points="';
// 		var at_start = true;
// 		for( let p of this.pts ) {
// 			const sp = transPt( S, p );
// 			if( at_start ) {
// 				at_start = false;
// 			} else {
// 				s = s + ' ';
// 			}
// 			s = s + `${sp.x},${sp.y}`;
// 		}
// 		const col = colmap[this.label];
//
// 		s = s + `" stroke="black" stroke-weight="0.1"
// fill="rgb(${col[0]},${col[1]},${col[2]})" />`; 		stream.push( s
// );
// 	}
// }
//
// class CurvyShape
// {
// 	constructor( pts, quad, label )
// 	{
// 		this.quad = quad;
// 		this.label = label;
//
// 		let blah = true;
//
// 		this.pts = [pts[pts.length-1]];
// 		for( const p of pts ) {
// 			const prev = this.pts[this.pts.length-1];
// 			const v = psub( p, prev );
// 			const w = pt( -v.y, v.x );
// 			if( blah ) {
// 				this.pts.push( pframe( prev, v, w, 0.33, 0.6 )
// ); 				this.pts.push( pframe( prev, v, w, 0.67,
// 0.6 )
// ); 			} else {
// this.pts.push( pframe( prev, v, w, 0.33, -0.6 ) );
// this.pts.push( pframe( prev, v, w, 0.67, -0.6 ) );
// 			}
// 			blah = !blah;
// 			this.pts.push( p );
// 		}
// 	}
//
// 	draw( S )
// 	{
// 		fill( ...colmap[this.label] );
// 		strokeWeight( 0.1 );
// 		stroke( 0 );
//
// 		beginShape();
// 		const tp = transPt( S, this.pts[0] );
// 		vertex( tp.x, tp.y );
//
// 		for( let idx = 1; idx < this.pts.length; idx += 3 ) {
// 			const a = transPt( S, this.pts[idx] );
// 			const b = transPt( S, this.pts[idx+1] );
// 			const c = transPt( S, this.pts[idx+2] );
//
// 			bezierVertex( a.x, a.y, b.x, b.y, c.x, c.y );
// 		}
// 		endShape( CLOSE );
// 	}
//
// 	streamSVG( S, stream )
// 	{
// 		const tp = transPt( S, this.pts[0] );
// 		vertex( tp.x, tp.y );
//
// 		var s = `<path d="M ${tp.x} ${tp.y}`;
//
// 		for( let idx = 1; idx < this.pts.length; idx += 3 ) {
// 			const a = transPt( S, this.pts[idx] );
// 			const b = transPt( S, this.pts[idx+1] );
// 			const c = transPt( S, this.pts[idx+2] );
//
// 			s = s + ` C ${a.x} ${a.y} ${b.x} ${b.y} ${c.x} ${c.y}`;
// 		}
// 		const col = colmap[this.label];
//
// 		s = s + `" stroke="black" stroke-weight="0.1"
// fill="rgb(${col[0]},${col[1]},${col[2]})" />`; 		stream.push( s
// );
// 	}
// }
//
// class Meta
// {
// 	constructor()
// 	{
// 		this.geoms = [];
// 		this.quad = [];
// 	}
//
// 	addChild( g, T )
// 	{
// 		this.geoms.push( { geom : g, xform: T } );
// 	}
//
// 	draw( S )
// 	{
// 		for( let g of this.geoms ) {
// 			g.geom.draw( mul( S, g.xform ) );
// 		}
// 	}
//
// 	streamSVG( S, stream )
// 	{
// 		for( let g of this.geoms ) {
// 			g.geom.streamSVG( mul( S, g.xform ), stream );
// 		}
// 	}
// }
//
// function buildHatTurtleBase( hat_dominant )
// {
// 	const r3 = 1.7320508075688772;
// 	const hr3 = 0.8660254037844386;
//
// 	function hexPt( x, y )
// 	{
// 		return pt( x + 0.5*y, -hr3*y );
// 	}
//
// 	function hexPt2( x, y )
// 	{
// 		return pt( x + hr3*y, -0.5*y );
// 	}
//
// 	const hat = [
// 		hexPt(-1, 2), hexPt(0, 2), hexPt(0, 3), hexPt(2, 2), hexPt(3,
// 0), 		hexPt(4, 0), hexPt(5,-1), hexPt(4,-2), hexPt(2,-1),
// hexPt(2,-2), hexPt( 1, -2), hexPt(0,-2), hexPt(-1,-1), hexPt(0, 0) ];
//
// 	const turtle = [
// 		hexPt(0,0), hexPt(2,-1), hexPt(3,0), hexPt(4,-1), hexPt(4,-2),
// 		hexPt(6,-3), hexPt(7,-5), hexPt(6,-5), hexPt(5,-4), hexPt(4,-5),
// 		hexPt(2,-4), hexPt(0,-3), hexPt(-1,-1), hexPt(0,-1)
// 		];
//
// 	const hat_keys = [
// 		hat[3], hat[5], hat[7], hat[11]
// 	];
// 	const turtle_keys = [
// 		turtle[3], turtle[5], turtle[7], turtle[11]
// 	];
//
// 	const ret = {};
//
// 	if( hat_dominant ) {
// 		for( lab of ['Delta', 'Theta', 'Lambda', 'Xi',
// 					 'Pi', 'Sigma', 'Phi', 'Psi'] ) {
// 			ret[lab] = new Shape( hat, hat_keys, lab );
// 		}
//
// 		const mystic = new Meta();
// 		mystic.addChild( new Shape( hat, hat_keys, 'Gamma1' ), ident );
// 		mystic.addChild( new Shape( turtle, turtle_keys, 'Gamma2' ),
// 			ttrans( hat[8].x, hat[8].y ) );
// 		mystic.quad = hat_keys;
// 		ret['Gamma'] = mystic;
// 	} else {
// 		for( lab of ['Delta', 'Theta', 'Lambda', 'Xi',
// 					 'Pi', 'Sigma', 'Phi', 'Psi'] ) {
// 			ret[lab] = new Shape( turtle, turtle_keys, lab );
// 		}
//
// 		const mystic = new Meta();
// 		mystic.addChild( new Shape( turtle, turtle_keys, 'Gamma1' ),
// ident ); 		mystic.addChild( new Shape( hat, hat_keys, 'Gamma2' ),
// mul( ttrans( turtle[9].x, turtle[9].y ), trot( PI/3 ) ) );
// mystic.quad = turtle_keys; 		ret['Gamma'] = mystic;
// 	}
//
// 	return ret;
// }
//
// function buildHexBase()
// {
// 	const hr3 = 0.8660254037844386;
//
// 	const hex = [
// 		pt(0, 0),
// 		pt(1.0, 0.0),
// 		pt(1.5, hr3),
// 		pt(1, 2*hr3),
// 		pt(0, 2*hr3),
// 		pt(-0.5, hr3)
// 	];
//
// 	const hex_keys = [ hex[1], hex[2], hex[3], hex[5] ];
//
// 	const ret = {};
//
// 	for( lab of ['Gamma', 'Delta', 'Theta', 'Lambda', 'Xi',
// 				 'Pi', 'Sigma', 'Phi', 'Psi'] ) {
// 		ret[lab] = new Shape( hex, hex_keys, lab );
// 	}
//
// 	return ret;
// }
//

// //
