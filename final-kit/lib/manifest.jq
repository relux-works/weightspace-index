{schema:2,task:$task,model_label:$label,endpoint:$url,shard_id:$shard,served_model:$model,protocol_id:$protocol,
 dataset:(if $task|startswith("mmlu_pro") then $pins[0].mmlu_pro_biology else $pins[0].minerva_math500 end | {path,name,split,revision}),
 sample_units:$units[0],selection_sha256:$selection,count:($units[0]|length),engine:"vllm",engine_version_expected:"0.33.0",plugin_expected:"vllm-metal==0.33.0",
 lineage:$lin[0],performed_by:"index",trust_level:"submitted",lm_eval_version:"0.4.13",sampling:$backend[0].generation_kwargs,
 config:{reference_config:$lin[0].reference_config,num_concurrent:12,servers:2,cli_batch_size:1,output_cap:32768},
 request_seed:$seed,harness_seeds:[0,1234,1234,$seed],source_sha256:$sources[0],utc_start:$time,utc_end:null,exit_status:null,state:"running",protocol_qualified:false,
 scoring:(if $task=="card-math" then "BLOCKED: upstream diagnostics only; manual last-box and cap review required" else "upstream lm_eval 0.4.13" end),
 finish_reason:null,prompt_tokens:null,completion_tokens:null,total_tokens:null,cap_hits:null,evidence_recovery:"manual: join sanitized vLLM response records by request and prompt identity; no automated recovery",authoritative_rescore:"pending: see RESCORE.md"}
