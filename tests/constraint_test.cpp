#include <catch2/catch_all.hpp>
#include <type_traits>

class MyTransFunc {
  public:
    using Cell = int;

    int operator()(int input) const { return input + 1; }

    int get_time_dependent_value() { return 17; }
};

// https://stackoverflow.com/questions/76511298/c-concept-requiring-certain-behavior-from-a-template
template <typename T>
concept DefinesTDV = requires(T t) { t.get_time_dependent_value(); };

template <typename TransFunc> class ExtendedTransFunc : public TransFunc {
  public:
    using Cell = typename TransFunc::Cell;

    int get_time_dependent_value()
        requires(!DefinesTDV<TransFunc>)
    {
        return 42;
    }

    int get_time_dependent_value()
        requires(DefinesTDV<TransFunc>)
    {
        return TransFunc::get_time_dependent_value();
    }
};

TEST_CASE("Constraint test", "[foo]") {
    ExtendedTransFunc<MyTransFunc> my_trans_func;
    REQUIRE(my_trans_func(2) == 3);
    REQUIRE(my_trans_func.get_time_dependent_value() == 17);
}