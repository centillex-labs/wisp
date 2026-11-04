use wisp_runtime::{
    tensor::{Tag, Tensor},
    value::Value,
    Interp,
};

const TRANSFORMER: &str =
    "n = transformer('vocab', {'a','b'}, 'ctx', 8, 'layers', 1, 'heads', 1, 'd', 2, 'ff', 2);\n";
fn interp(budget: u64) -> Interp {
    let mut it = Interp::new("observation.wisp", "");
    it.echo = false;
    it.harness.passes = Some(budget);
    it
}
fn run(it: &mut Interp, src: &str) -> Result<wisp_runtime::interp::Env, wisp_syntax::Diag> {
    it.src = std::rc::Rc::from(src);
    it.run(&wisp_syntax::parse(src).unwrap())
}
fn register(it: &mut Interp, name: &str, values: &[f64]) {
    it.harness
        .register_dataset(name, Value::Arr(Tensor::col(values.to_vec())))
        .unwrap();
}
fn texts(it: &mut Interp, name: &str, values: &[&str]) {
    it.harness
        .register_dataset(name, Value::cell(values.iter().map(Value::str).collect()))
        .unwrap();
}

#[test]
fn teacher_forcing_returns_per_token_logprobs_and_hidden_states() {
    let mut it = interp(1);
    let src=format!("{TRANSFORMER}\nn.embed = eye(2); n.unembed=eye(2);\n[lp, h] = tf(n, {{'aab'}});\nassert(abs(lp{{0}}[0] + log(1+exp(-1))) < 1e-12);\nassert(abs(lp{{0}}[1] + 1 + log(1+exp(-1))) < 1e-12);\nassert(h{{0}} == [1 0; 1 0; 0 1]);\n");
    let env = run(&mut it, &src).unwrap();
    let Value::Cell(lp) = &env["lp"] else {
        panic!()
    };
    let Value::Arr(lp) = &lp[0] else { panic!() };
    assert_eq!(lp.tag, Tag::Act);
    assert_eq!(lp.shape, vec![1, 2]);
    assert_eq!(it.harness.used, 1);
}

#[test]
fn attention_is_causal_normalized_and_observation_tagged() {
    let mut it = interp(2);
    let src=format!("{TRANSFORMER}\nmaps=attn(n, {{'aab', 'ab'}}); a=maps{{0}};\nassert(size(a, 0)==1); assert(size(a, 1)==1); assert(size(a, 2)==3);\nassert(a[0,0,0,0]==1); assert(a[0,0,0,1]==0);\nassert(a[0,0,1,0]==0.5); assert(a[0,0,1,1]==0.5); assert(a[0,0,1,2]==0);\nassert(abs(a[0,0,2,0]-1/3)<1e-12);\n");
    let env = run(&mut it, &src).unwrap();
    let Value::Arr(a) = &env["a"] else { panic!() };
    assert_eq!(a.tag, Tag::Act);
    assert_eq!(it.harness.used, 2);
}

#[test]
fn generation_is_seeded_greedy_or_sampled_with_optional_grading() {
    let mut it = interp(3);
    let src=format!("{TRANSFORMER}\nn.embed[0,:]=1; n.unembed[1,0]=3;\ng=gen(n, {{'a','b'}}, 3, 'temperature', 0, 'grade', @(s) count(s,'b'));\nassert(g.texts{{0}}=='bbb'); assert(g.texts{{1}}=='bbb'); assert(g.scores==[3;3]);\na=gen(n, {{'a'}}, 5, 'seed', 12); b=gen(n, {{'a'}}, 5, 'seed', 12);\nassert(a.tokens{{0}}==b.tokens{{0}});\n");
    let env = run(&mut it, &src).unwrap();
    let Value::Struct(g) = &env["g"] else {
        panic!()
    };
    let Value::Arr(scores) = &g["scores"] else {
        panic!()
    };
    assert_eq!(scores.tag, Tag::Act);
    assert_eq!(it.harness.used, 3);
}

#[test]
fn batch_costs_round_up_and_charge_before_work() {
    for (function, n, extra) in [("tf", 1025, ""), ("gen", 513, ", 1")] {
        let mut it = interp(2);
        let seqs = Value::cell((0..n).map(|_| Value::str("ab")).collect());
        it.harness.register_dataset("batch", seqs).unwrap();
        run(
            &mut it,
            &format!("{TRANSFORMER}\nx={function}(n,'batch'{extra});"),
        )
        .unwrap();
        assert_eq!(it.harness.used, 2);
        assert_eq!(it.harness.log[0]["cost"], 2);
        let mut refused = interp(1);
        texts(&mut refused, "batch", &vec!["ab"; n]);
        let err = run(
            &mut refused,
            &format!("{TRANSFORMER}\nx={function}(n,'batch'{extra});"),
        )
        .err()
        .unwrap();
        assert_eq!(err.code, "E-BUDGET");
        assert_eq!(refused.harness.used, 0);
        assert!(refused.harness.observed.is_empty());
        assert_eq!(refused.harness.log[0]["refused"], true);
    }
}

#[test]
fn pair_and_confirm_use_paired_statistics_and_one_outer_charge() {
    let mut it = interp(2);
    register(&mut it, "train", &[0., 1., 2.]);
    register(&mut it, "fresh", &[3., 4., 5.]);
    let env = run(
        &mut it,
        r#"
a=mlp([1 1],'relu'); b=copy(a); b.b1[0]=1;
metric=@(m,items) run(m,items);
[g,se]=pair(a,b,'train',metric); assert(g==1); assert(se==0);
ok=confirm(a,b,'fresh',metric); assert(ok); assert(budget()==0);
"#,
    )
    .unwrap();
    let Value::Arr(ok) = &env["ok"] else { panic!() };
    assert_eq!(ok.tag, Tag::Act);
    assert_eq!(it.harness.used, 2);
    assert_eq!(it.harness.log.iter().filter(|x| x["cost"] == 1).count(), 2);
    assert_eq!(it.harness.log.iter().filter(|x| x["cost"] == 0).count(), 4);
}

#[test]
fn confirm_rejects_observed_data_including_renamed_and_inline_copies() {
    for source in [
        "run(a,'train');",
        "run(a,[0;1;2]);",
        "pair(a,a,'train',metric);",
        "p=mlp([3 1],'relu'); q=p.W1*[0;1;2];",
    ] {
        let mut it = interp(5);
        register(&mut it, "train", &[0., 1., 2.]);
        register(&mut it, "alias", &[0., 1., 2.]);
        let err=run(&mut it,&format!("a=mlp([1 1],'relu'); metric=@(m,s) run(m,s); {source} confirm(a,a,'alias',metric);")).err().unwrap();
        assert_eq!(err.code, "E-REUSED-SET");
        assert!(err.message.contains("at observation.wisp:"));
        assert_eq!(it.harness.used, 1);
    }
}

#[test]
fn all_transformer_observations_record_named_set_exposure() {
    for (name, suffix, cost) in [("tf", "", 1), ("attn", "", 2), ("gen", ", 1", 1)] {
        let mut it = interp(5);
        texts(&mut it, "texts", &["ab", "ba"]);
        texts(&mut it, "renamed", &["ab", "ba"]);
        let err=run(&mut it,&format!("{TRANSFORMER}\nx={name}(n,'texts'{suffix}); confirm(n,n,'renamed',@(m,s) run(m,s)[:,0]);")).err().unwrap();
        assert_eq!(err.code, "E-REUSED-SET");
        assert!(err.message.contains(name));
        assert_eq!(it.harness.used, cost);
    }
}

#[test]
fn hidden_test_sets_are_refused_by_name_alias_and_inline_content() {
    for input in ["'test'", "'alias'", "[8;9]"] {
        let mut it = interp(4);
        register(&mut it, "test", &[8., 9.]);
        register(&mut it, "alias", &[8., 9.]);
        it.harness.test_sets.push("test".into());
        let err = run(&mut it, &format!("a=mlp([1 1],'relu'); run(a,{input});"))
            .err()
            .unwrap();
        assert_eq!(err.code, "E-FORBIDDEN");
        assert_eq!(it.harness.used, 0);
        assert!(it.harness.log.is_empty());
    }
}

#[test]
fn paired_metrics_cannot_substitute_data_or_fabricate_unobserved_scores() {
    for (metric, code) in [
        ("@(m,s) run(m,[9;10;11])", "E-FORBIDDEN"),
        ("@(m,s) ones(3,1)", "E-FORBIDDEN"),
        ("@(m,s) sum(run(m,s))", "E-SHAPE"),
    ] {
        let mut it = interp(5);
        register(&mut it, "fresh", &[0., 1., 2.]);
        let err = run(
            &mut it,
            &format!("a=mlp([1 1],'relu'); confirm(a,a,'fresh',{metric});"),
        )
        .err()
        .unwrap();
        assert_eq!(err.code, code);
        assert!(it.harness.covered_by.is_none());
        assert!(it.harness.required_data.is_none());
        assert_eq!(it.harness.used, 1);
        run(&mut it, "a=mlp([1 1],'relu'); run(a,[20;21;22]);").unwrap();
        assert_eq!(
            it.harness.used, 2,
            "failed callback must not leave observations free"
        );
    }
}

#[test]
fn failed_confirm_does_not_run_the_second_metric_and_consumes_the_set() {
    let mut it = interp(3);
    register(&mut it, "fresh", &[0., 1., 2.]);
    let err = run(
        &mut it,
        r#"
function s = metric(m,items)
    s=run(m,items);
    error('stop');
end
a=mlp([1 1],'relu'); confirm(a,a,'fresh',@metric);
"#,
    )
    .err()
    .unwrap();
    assert_eq!(err.code, "E-USER");
    assert_eq!(it.harness.log.len(), 2);
    assert!(it.harness.covered_by.is_none());
    let err = run(
        &mut it,
        "a=mlp([1 1],'relu'); confirm(a,a,'fresh',@(m,s) run(m,s));",
    )
    .err()
    .unwrap();
    assert_eq!(err.code, "E-REUSED-SET");
}

#[test]
fn budget_refusal_leaves_confirmation_set_fresh() {
    let mut it = interp(0);
    register(&mut it, "fresh", &[0., 1., 2.]);
    let src =
        "a=mlp([1 1],'relu'); b=copy(a); b.b1[0]=1; assert(confirm(a,b,'fresh',@(m,s) run(m,s)));";
    assert_eq!(run(&mut it, src).err().unwrap().code, "E-BUDGET");
    assert!(it.harness.observed.is_empty());
    it.harness.passes = Some(1);
    run(&mut it, src).unwrap();
    assert_eq!(it.harness.used, 1);
}
