def request: del(.utc_start, .utc_end, .exit_status, .state);
length == 1 and (.[0] |
  .state == "complete-unqualified" and .exit_status == 0 and
  request == ($requested[0] | request))
