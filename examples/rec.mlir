builtin.module {
  "aziz.func"() ({
    %0 = "aziz.string_constant"() {value = "hello world!"} : () -> #aziz.string
    "aziz.return"(%0) : (#aziz.string) -> ()
  }) {sym_name = "get_msg", function_type = () -> #aziz.string} : () -> ()
  "aziz.func"() ({
  ^bb0(%1 : i32):
    %2 = "aziz.constant"() {value = 1 : i32} : () -> i32
    %3 = "aziz.le"(%1, %2) : (i32, i32) -> i32
    %4 = "aziz.if"(%3) ({
      %5 = "aziz.constant"() {value = 1 : i32} : () -> i32
      "aziz.yield"(%5) : (i32) -> ()
    }, {
      %6 = "aziz.constant"() {value = 1 : i32} : () -> i32
      %7 = "aziz.sub"(%1, %6) : (i32, i32) -> i32
      %8 = "aziz.call"(%7) {callee = @factorial} : (i32) -> i32
      %9 = "aziz.mul"(%1, %8) : (i32, i32) -> i32
      "aziz.yield"(%9) : (i32) -> ()
    }) : (i32) -> i32
    "aziz.return"(%4) : (i32) -> ()
  }) {sym_name = "factorial", function_type = (i32) -> i32} : () -> ()
  "aziz.func"() ({
    %10 = "aziz.call"() {callee = @get_msg} : () -> #aziz.string
    "aziz.print"(%10) : (#aziz.string) -> ()
    %11 = "aziz.constant"() {value = 5 : i32} : () -> i32
    %12 = "aziz.call"(%11) {callee = @factorial} : (i32) -> i32
    "aziz.print"(%12) : (i32) -> ()
    %13 = "aziz.constant"() {value = 0 : i32} : () -> i32
    "aziz.return"(%13) : (i32) -> ()
  }) {sym_name = "main", function_type = () -> i32} : () -> ()
} 
