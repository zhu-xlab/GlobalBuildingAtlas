%module bilateralfilter
%include "std_vector.i"


%{
#define SWIG_FILE_WITH_INIT
#include "bilateralfilter.hpp"
#include "permutohedral.hpp"
%}

namespace std {
  %template(FloatVector) vector<float>;
}

%include "numpy.i"

%init %{
    import_array();
%}

%apply (float* IN_ARRAY1, int DIM1) {(float* image, int len_image),(float* in, int len_in)}
%apply (float* INPLACE_ARRAY1, int DIM1) {(float * out, int len_out)}

%apply (float* IN_ARRAY1, int DIM1) {(float* images, int len_images),(float* ins, int len_ins)}
%apply (float* INPLACE_ARRAY1, int DIM1) {(float * outs, int len_outs)}

%include "bilateralfilter.hpp"
%include "permutohedral.hpp"
