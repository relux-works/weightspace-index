# Map frozen global rows into the upstream category-filtered child index spaces.
def fail($m): error($m);
($pins[0]) as $p | ($spec[0]) as $s | ($strata[0]) as $st |
(if $task=="card-math" then "minerva_math500" elif $task|startswith("mmlu_pro_") then "mmlu_pro" else $task end) as $parent |
(if $parent=="mmlu_pro" then $p.mmlu_pro_biology else $p.minerva_math500 end) as $pin |
if ($spec|length) != 1 or ($s.indices|type) != "array" or $s.task != $task or $s.dataset != $pin.path or $s.revision != $pin.revision then fail("selection identity mismatch") else . end |
$s.indices as $all |
if ($all|length)==0 or ($all|unique|length)!=($all|length) or (all($all[];type=="number" and floor==. and .>=0)|not) then fail("invalid/duplicate indices") else . end |
(if $parent=="mmlu_pro" then
  reduce $st.mmlu_pro[] as $r ({counts:{},rows:[]};
    ($r.category|gsub(" ";"_")) as $c | (.counts[$c]//0) as $i |
    .rows += [{global_index:$r.index,task:("mmlu_pro_"+$c),index:$i,stratum:$r.category}] | .counts[$c]=$i+1) | .rows
 else $st.minerva_math500 | map({global_index:.index,task:(if $task=="card-math" then "card_math" else "minerva_math500" end), index:.index,stratum:(.subject+"|level="+(.level|tostring))}) end) as $map |
(if $task|startswith("mmlu_pro_") then [$all[] as $i | $map[] | select(.task==$task and .index==$i)]
 else [$all[] as $i | $map[$i] | select(.!=null)] end) as $units |
if ($units|length)!=($all|length) then fail("index out of range") else . end |
$units[$start:$start+$count] as $chosen |
if ($chosen|length)!=$count or $count<1 then fail("selection range incomplete") else $chosen end
