def demand($ok;$s): if $ok then . else error($s) end;
def identity: {task,dataset,protocol_id,config,engine,lm_eval_version,sampling,source_sha256,
    base_weights_digest:.lineage.base_weights_digest,tokenizer_digest:.lineage.tokenizer_digest,
    template_digest:.lineage.template_digest,quantization:.lineage.quantization,
    conversion_tool_versions:.lineage.conversion_tool_versions,
    engine_version_observed:.lineage.engine_version_observed,plugin_version_observed:.lineage.plugin_version_observed,
    machine_layout_evidence:.lineage.machine_layout_evidence};
def key: [.task,.index];
def selection: map([.task,.index,.global_index,.stratum]) | sort;
def side($rows):
  $rows[0].manifest as $m |
  demand(all($rows[]; .manifest.task==$spec[0].task and .manifest.dataset.path==$spec[0].dataset and .manifest.dataset.revision==$spec[0].revision);"selection identity mismatch") |
  demand(all($rows[]; (.manifest|identity)==($m|identity) and .manifest.model_label==$m.model_label and .manifest.lineage==$m.lineage and .manifest.served_model==$m.served_model and .manifest.request_seed==$m.request_seed);"mixed config within arm") |
  [$rows[] | . as $bundle | .items[] | . as $i |
    [$bundle.native[]|select(.native_task==$i.task and .doc_id==$i.index)] as $native |
    demand(($native|length)==1;"native item missing/duplicate") |
    $native[0] as $r |
    demand($r.prompt_hash==$i.prompt_sha256 and $r.doc_hash==$i.doc_sha256;"native hashes mismatch") |
    (if $i.task=="minerva_math500" then {exact_match:$r.exact_match,math_verify:$r.math_verify} else {exact_match:$r.exact_match} end) as $scores |
    demand(all($scores[]; .==0 or .==1);"invalid upstream scores") |
    $i+{scores:$scores}] as $items |
  demand(($items|selection)==($expected[0]|selection);"missing/unexpected/duplicate paired units") |
  demand(all($rows[]; (.items|selection)==(.manifest.sample_units|selection) and (.native|length)==(.items|length));"manifest/native coverage mismatch") |
  {manifest:$m,items:($items|sort_by(key)),shards:($rows|length)};
side($a) as $x | side($b) as $y |
demand($x.manifest.model_label!=$y.manifest.model_label;"distinct arm labels required") |
demand(($x.manifest|identity)==($y.manifest|identity);"protocol/config mismatch between sides") |
demand(if $aa then $x.manifest.lineage==$y.manifest.lineage and $x.manifest.served_model==$y.manifest.served_model and $x.manifest.request_seed!=$y.manifest.request_seed else $x.manifest.request_seed==$y.manifest.request_seed end;"paired seed/A-A lineage mismatch") |
demand(($x.items|map([.task,.index,.prompt_sha256,.doc_sha256]))==($y.items|map([.task,.index,.prompt_sha256,.doc_sha256]));"paired prompt/doc mismatch") |
# Both sides have just been sorted and checked for identical item identities.
[range($x.items|length) as $i | $x.items[$i] |
  {task,index,global_index,stratum,a:.scores,b:$y.items[$i].scores}] as $pairs |
{task:$spec[0].task,protocol_id:$x.manifest.protocol_id,status:"DESCRIPTIVE frozen selection; deployment unqualified; authoritative re-scoring pending",output_cap:32768,aa_control:$aa,
 expected:($pairs|length),models:[$x.manifest.model_label,$y.manifest.model_label],shards:[$x.shards,$y.shards],
 metrics:[$pairs[0].a|keys[] as $metric |
   {metric:$metric,a_correct:([$pairs[].a[$metric]]|add),b_correct:([$pairs[].b[$metric]]|add),denominator:($pairs|length),
   discordant:([$pairs[]|select(.a[$metric]!=.b[$metric])]|length),wins:([$pairs[]|select(.a[$metric]==0 and .b[$metric]==1)]|length),losses:([$pairs[]|select(.a[$metric]==1 and .b[$metric]==0)]|length)}],pairs:$pairs,
 weighted_tier_summary:(if $spec[0].stratum_population then
   ($spec[0].stratum_population) as $pop |
   [$pairs|group_by(.stratum)[]|.[0].stratum as $h |
     {stratum:$h,n:length,expected_n:$spec[0].stratum_sample[$h],weight:($pop[$h]/$spec[0].population),
      a:([.[].a.exact_match]|add/length),b:([.[].b.exact_match]|add/length)}] as $hs |
   if (all($hs[]; .n==.expected_n and .weight!=null)) and ($hs|length)==($pop|length) then
     {metric:"exact_match",a:([$hs[]|.weight*.a]|add),b:([$hs[]|.weight*.b]|add),strata:$hs}
   else error("incomplete frozen strata; no tier score") end
 else null end)}
