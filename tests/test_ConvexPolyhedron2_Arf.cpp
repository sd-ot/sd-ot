#include "../src/sdot/Support/Stream.h"
#include "../src/sdot/Geometry/ConvexPolyhedron2.h"
#include "../src/sdot/Display/VtkOutput.h"
#include "./catch_main.h"
#include <iostream>

//// nsmake cpp_flag -march=native

using namespace sdot;

//template<class LC>
//void test_bnd( LC &icp, FunctionEnum::Arfd &arf, std::set<double> values ) {
//    std::set<double> exp;

//    icp.for_each_boundary_item( arf, [&]( const auto &bi ) {
//        exp.insert( bi.measure );
//    }, 10 );

//    REQUIRE( exp.size() == values.size() );
//    for( auto a = exp.begin(), b = values.begin(); a != exp.begin(); ++a, ++b )
//        CHECK_THAT( *a, WithinAbs( *b, 1e-6 ) );
//}

//TEST_CASE( "Arf", "p_1" ) {
//    struct Pc { enum { dim = 2, allow_ball_cut = 1 }; using TI = std::size_t; using TF = double; using CI = std::string; };
//    using  LC = ConvexPolyhedron2<Pc>;
//    using  TF = LC::TF;
//    using  std::sqrt;
//    using  std::pow;

//    // box { 0, 0 }, { 2, 1 }
//    LC icp( LC::Box{ { 0, 0 }, { 2, 1 } } );
//    icp.sphere_center = { 0, 0 };

//    //
//    FunctionEnum::Arfd arf;
//    arf.values = []( double r ) -> double {
//        return r < 1 ? 1 - r * r : 0;
//    };
//    arf.inp_scaling = []( TF w ) -> TF {
//        return TF( 1 ) / sqrt( w );
//    };
//    arf.out_scaling = []( TF w ) -> TF {
//        return w;
//    };
//    arf.stops = { 1.0 };

//    // integrations
//    // Integrate[ Integrate[ 100 - ( x * x + y * y ), { x, 0, 2 } ], { y, 0, 1 } ] => 590.0 / 3.0
//    // Integrate[ Integrate[ 10 - ( x * x + y * y ), { x, 0, 2 } ], { y, 0, 1 } ] => 50.0 / 3.0
//    // Integrate[ Integrate[ ( 1   - ( x * x + y * y ) ) * UnitStep[ 1   - x^2 - y^2 ], { x, 0, 1 } ], { y, 0, 1 } ] => 50.0 / 3.0
//    // Integrate[ Integrate[ ( 1/2 - ( x * x + y * y ) ) * UnitStep[ 1/2 - x^2 - y^2 ], { x, 0, 1 } ], { y, 0, 1 } ] => M_PI / 32
//    CHECK_THAT( icp.integration( arf, 100 ), WithinAbs( 590.0 /  3, 1e-6 ) );
//    CHECK_THAT( icp.integration( arf,  10 ), WithinAbs(  50.0 /  3, 1e-6 ) );
//    CHECK_THAT( icp.integration( arf,   1 ), WithinAbs(  M_PI /  8, 1e-6 ) );
//    CHECK_THAT( icp.integration( arf, 0.5 ), WithinAbs(  M_PI / 32, 1e-6 ) );


//    /*
//     centroids

//     c[ f_ ] := { Integrate[ Integrate[ x * f, { x, 0, 2 } ], { y, 0, 1 } ] / Integrate[ Integrate[ f, { x, 0, 2 } ], { y, 0, 1 } ], Integrate[ Integrate[ y * f, { x, 0, 2 } ], { y, 0, 1 } ] / Integrate[ Integrate[ f, { x, 0, 2 } ], { y, 0, 1 } ] }
//     m[ w_ ] := c[ ( w - ( x * x + y * y ) ) * UnitStep[ w - ( x * x + y * y ) ] ]
//     m[ 100 ]
//     m[  10 ]
//     m[   1 ]
//     m[ 1/2 ]
//     */
//    CHECK_THAT( icp.centroid( arf, 100 )[ 0 ], WithinAbs( 293.0 /  295                        , 1e-6 ) );
//    CHECK_THAT( icp.centroid( arf, 100 )[ 1 ], WithinAbs( 589.0 / 1180                        , 1e-6 ) );
//    CHECK_THAT( icp.centroid( arf,  10 )[ 0 ], WithinAbs(  23.0 /   25                        , 1e-6 ) );
//    CHECK_THAT( icp.centroid( arf,  10 )[ 1 ], WithinAbs(  49.0 /  100                        , 1e-6 ) );
//    CHECK_THAT( icp.centroid( arf,   1 )[ 0 ], WithinAbs( 16 / ( 15 * M_PI )                  , 1e-6 ) );
//    CHECK_THAT( icp.centroid( arf,   1 )[ 1 ], WithinAbs( 16 / ( 15 * M_PI )                  , 1e-6 ) );
//    CHECK_THAT( icp.centroid( arf, 0.5 )[ 0 ], WithinAbs( 8 * std::sqrt( 2.0 ) / ( 15 * M_PI ), 1e-6 ) );
//    CHECK_THAT( icp.centroid( arf, 0.5 )[ 1 ], WithinAbs( 8 * std::sqrt( 2.0 ) / ( 15 * M_PI ), 1e-6 ) );

//    /*
//      y = 0; Integrate[ 10 - ( x * x + y * y ), { x, 0, 2 } ] => 52.0/3
//      y = 1; Integrate[ 10 - ( x * x + y * y ), { x, 0, 2 } ] => 46.0/3
//      x = 0; Integrate[ 10 - ( x * x + y * y ), { y, 0, 1 } ] => 29.0/3
//      x = 2; Integrate[ 10 - ( x * x + y * y ), { y, 0, 1 } ] => 17.0/3
//    */
//    test_bnd( icp, arf, { 52.0/3, 46.0/3, 29.0/3, 17.0/3 } );
//}

TEST_CASE( "Arf_2_5", "p_2_5" ) {
    struct Pc { enum { dim = 2, allow_ball_cut = 0 }; using TI = std::size_t; using TF = double; using CI = std::string; };
    using  LC = ConvexPolyhedron2<Pc>;
    using  TF = LC::TF;
    using  std::sqrt;
    using  std::pow;

    // box { 0, 0 }, { 2, 1 }
    LC icp( LC::Box{ { 0, 0 }, { 2, 1 } } );
    icp.sphere_center = { 0, 0 };

    //
    FunctionEnum::Arfd arf;
    arf.values = []( double r ) -> double {
        return r < 1 ? pow( 1 - r * r, 2.5 ) : 0;
    };
    arf.inp_scaling = []( TF w ) -> TF {
        return pow( w, - 0.5 );
    };
    arf.out_scaling = []( TF w ) -> TF {
        return pow( w, 2.5 );
    };
    arf.stops = { 1.0 };
    arf.prec = 1e-7;

    arf.make_approximations_if_not_done();

    CHECK_THAT( icp.integration( arf, 100 ), WithinAbs( 191826.6660609519, 1e-4 ) );
    CHECK_THAT( icp.integration( arf,  10 ), WithinAbs(    417.04        , 1e-2 ) );
}
