if(NOT DEFINED CNN_EXECUTABLE OR NOT DEFINED TEST_OUTPUT_DIR)
  message(FATAL_ERROR "CNN_EXECUTABLE and TEST_OUTPUT_DIR are required")
endif()

file(REMOVE_RECURSE "${TEST_OUTPUT_DIR}/cnn_cli")
set(root "${TEST_OUTPUT_DIR}/cnn_cli")
file(MAKE_DIRECTORY "${root}/empty" "${root}/unknown/mystery"
     "${root}/wrongsize/yes")
file(WRITE "${root}/unknown/mystery/sample.bin" "0 1 2")
file(WRITE "${root}/wrongsize/yes/sample.bin" "0 1 2")
set(model "${root}/model.bin")

function(expect_failure name)
  execute_process(COMMAND "${CNN_EXECUTABLE}" ${ARGN}
    RESULT_VARIABLE result OUTPUT_QUIET ERROR_QUIET)
  if("${result}" STREQUAL "0")
    message(FATAL_ERROR "${name}: command unexpectedly succeeded")
  endif()
endfunction()

execute_process(COMMAND "${CNN_EXECUTABLE}" --help
  RESULT_VARIABLE result OUTPUT_QUIET ERROR_QUIET)
if(NOT "${result}" STREQUAL "0")
  message(FATAL_ERROR "--help should succeed (result: ${result})")
endif()

expect_failure(bad-arguments train "${root}/empty" "${model}" --epochs 0)
expect_failure(missing-dataset train "${root}/missing" "${model}" --epochs 1)
expect_failure(empty-dataset train "${root}/empty" "${model}" --epochs 1)
expect_failure(unknown-label train "${root}/unknown" "${model}" --epochs 1)
expect_failure(wrong-sample-size train "${root}/wrongsize" "${model}" --epochs 1)
expect_failure(missing-model-argument score "${root}/empty")
