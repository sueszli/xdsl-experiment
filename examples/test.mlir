builtin.module {
  "aziz.func"() ({
  ^bb0(%0 : i32, %1 : i32):
    %2 = "aziz.add"(%0, %1) : (i32, i32) -> i32
    "aziz.return"(%2) : (i32) -> ()
  }) {sym_name = "add_ints", function_type = (i32, i32) -> i32} : () -> ()
  "aziz.func"() ({
  ^bb1(%3 : i32, %4 : i32):
    %5 = "aziz.add"(%3, %4) : (i32, i32) -> i32
    "aziz.return"(%5) : (i32) -> ()
  }) {sym_name = "add_floats", function_type = (i32, i32) -> i32} : () -> ()
  "aziz.func"() ({
  ^bb2(%6 : i32, %7 : i32):
    %8 = "aziz.mul"(%6, %7) : (i32, i32) -> i32
    "aziz.return"(%8) : (i32) -> ()
  }) {sym_name = "multiply_floats", function_type = (i32, i32) -> i32} : () -> ()
  "aziz.func"() ({
    %9 = "aziz.string_constant"() {value = "hello"} : () -> #aziz.string
    "aziz.return"(%9) : (#aziz.string) -> ()
  }) {sym_name = "get_msg", function_type = () -> #aziz.string} : () -> ()
  "aziz.func"() ({
    %10 = "aziz.constant"() {value = 10 : i32} : () -> i32
    %11 = "aziz.constant"() {value = 20 : i32} : () -> i32
    %12 = "aziz.call"(%10, %11) {callee = @add_ints} : (i32, i32) -> i32
    "aziz.print"(%12) : (i32) -> ()
    %13 = "aziz.constant"() {value = 3.140000e+00 : f64} : () -> f64
    %14 = "aziz.constant"() {value = 2.860000e+00 : f64} : () -> f64
    %15 = "aziz.call"(%13, %14) {callee = @add_floats} : (f64, f64) -> i32
    "aziz.print"(%15) : (i32) -> ()
    %16 = "aziz.constant"() {value = 1.500000e+00 : f64} : () -> f64
    %17 = "aziz.constant"() {value = 4.000000e+00 : f64} : () -> f64
    %18 = "aziz.call"(%16, %17) {callee = @multiply_floats} : (f64, f64) -> i32
    "aziz.print"(%18) : (i32) -> ()
    %19 = "aziz.constant"() {value = 2.500000e+00 : f64} : () -> f64
    %20 = "aziz.constant"() {value = 4.000000e+00 : f64} : () -> f64
    %21 = "aziz.mul"(%19, %20) : (f64, f64) -> f64
    "aziz.print"(%21) : (f64) -> ()
    %22 = "aziz.constant"() {value = 1.500000e+00 : f64} : () -> f64
    %23 = "aziz.constant"() {value = 2.500000e+00 : f64} : () -> f64
    %24 = "aziz.add"(%22, %23) : (f64, f64) -> f64
    "aziz.print"(%24) : (f64) -> ()
    %25 = "aziz.string_constant"() {value = "String with (parens) inside"} : () -> #aziz.string
    "aziz.print"(%25) : (#aziz.string) -> ()
    %26 = "aziz.string_constant"() {value = "(more"} : () -> #aziz.string
    "aziz.print"(%26) : (#aziz.string) -> ()
    %27 = "aziz.string_constant"() {value = "data)"} : () -> #aziz.string
    "aziz.print"(%27) : (#aziz.string) -> ()
    %28 = "aziz.call"() {callee = @get_msg} : () -> #aziz.string
    "aziz.print"(%28) : (#aziz.string) -> ()
    %29 = "aziz.constant"() {value = 0 : i32} : () -> i32
    "aziz.return"(%29) : (i32) -> ()
  }) {sym_name = "main", function_type = () -> i32} : () -> ()
} 
