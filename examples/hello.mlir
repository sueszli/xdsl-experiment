builtin.module {
  "aziz.func"() ({
  ^bb0(%0 : i32):
    "aziz.return"(%0) : (i32) -> ()
  }) {sym_name = "greet", function_type = (i32) -> i32} : () -> ()
  "aziz.func"() ({
    %1 = "aziz.string_constant"() {value = "Hello, World!"} : () -> !aziz.string
    %2 = "aziz.call"(%1) {callee = @greet} : (!aziz.string) -> i32
    "aziz.return"(%2) : (i32) -> ()
  }) {sym_name = "main", function_type = () -> i32} : () -> ()
} 
