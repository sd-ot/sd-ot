#pragma once

#include "WmR2.h"

namespace sdot {
namespace FunctionEnum {

/// pos_part( w - r * r )
struct PpWmR2 {
    template<class PT,class TF>
    auto operator()( PT p, PT c, TF w ) const {
        auto r2 = norm_2_p2( p - c );
        return ( w - r2 ) * ( r2 <= w );
    }

    const char *name() const {
        return "PpWmR2";
    }

    auto func_for_final_cp_integration() const {
        return WmR2{};
    }

    N<1> need_ball_cut() const {
        return {};
    }

    template<class TF,class TS>
    void span_for_viz( const TF&, TS ) const {}
};

}
}
