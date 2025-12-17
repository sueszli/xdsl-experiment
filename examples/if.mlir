builtin.module {
  "aziz.func"() ({
  ^bb0(%0 : i32):
    %1 = "aziz.if"(%0) ({
      %2 = "aziz.constant"() {value = 1 : i32} : () -> i32
      "aziz.yield"(%2) : (i32) -> ()
    }, {
      %3 = "aziz.constant"() {value = 0 : i32} : () -> i32
      "aziz.yield"(%3) : (i32) -> ()
    }) : (i32) -> i32
    "aziz.return"(%1) : (i32) -> ()
  }) {sym_name = "test", function_type = (i32) -> i32} : () -> ()
  "aziz.func"() ({
    %4 = "aziz.constant"() {value = 1 : i32} : () -> i32
    %5 = "aziz.call"(%4) {callee = @test} : (i32) -> i32
    "aziz.print"(%5) : (i32) -> ()
    %6 = "aziz.constant"() {value = 0 : i32} : () -> i32
    "aziz.return"(%6) : (i32) -> ()
  }) {sym_name = "main", function_type = () -> i32} : () -> ()
} 
