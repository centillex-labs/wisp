use wisp_runtime::Interp;

fn interpreter() -> Interp {
    let mut it = Interp::new("self-contained.wisp", "");
    it.echo = false;
    it
}
fn run(it: &mut Interp, source: &str) -> Result<wisp_runtime::interp::Env, wisp_syntax::Diag> {
    it.src = std::rc::Rc::from(source);
    it.run(&wisp_syntax::parse(source).unwrap())
}

#[test]
fn inline_datasets_support_observation_and_fresh_confirmation() {
    let mut it = interpreter();
    run(&mut it, "budget(2); X=[0;1;2]; dataset('practice',X); X[0]=99; dataset('fresh',[3;4;5]); a=mlp([1 1],'relu'); a.W1[0,0]=1; b=copy(a); b.b1[0]=1; y=run(a,'practice'); assert(y==[0;1;2]); assert(confirm(a,b,'fresh',@(m,s) run(m,s))); assert(budget()==0);").unwrap();
    assert_eq!(it.harness.used, 2);
}

#[test]
fn budget_can_tighten_but_never_raise_or_reset_a_limit() {
    let mut it = interpreter();
    run(&mut it, "budget(3); a=mlp([1 1],'relu'); run(a,[1]); assert(budget()==2); budget(100); assert(budget()==2); budget(2); assert(budget()==1); run(a,[2]); budget(100); assert(budget()==0);").unwrap();
    assert_eq!(it.harness.used, 2);
    assert_eq!(it.harness.passes, Some(2));
    let error = run(&mut it, "budget(1);").err().unwrap();
    assert_eq!(error.code, "E-BUDGET");
    assert_eq!(it.harness.passes, Some(2));
    assert_eq!(it.harness.used, 2);
}

#[test]
fn host_budget_is_a_ceiling_for_inline_settings() {
    let mut it = interpreter();
    it.harness.passes = Some(0);
    let error = run(
        &mut it,
        "budget(100); assert(budget()==0); a=mlp([1 1],'relu'); run(a,[1]);",
    )
    .err()
    .unwrap();
    assert_eq!(error.code, "E-BUDGET");
    assert_eq!(it.harness.used, 0);
}

#[test]
fn dataset_aliases_cannot_reset_confirmation_freshness() {
    let mut it = interpreter();
    let error = run(&mut it, "budget(3); dataset('a',[1;2]); dataset('alias',[1;2]); n=mlp([1 1],'relu'); run(n,'a'); confirm(n,n,'alias',@(m,s) run(m,s));").err().unwrap();
    assert_eq!(error.code, "E-REUSED-SET");
    assert_eq!(it.harness.used, 1);
    assert_eq!(
        run(&mut it, "dataset('a',[3;4]);").err().unwrap().code,
        "E-DATASET"
    );
}

#[test]
fn inline_protection_is_enforced_before_saving() {
    let mut it = interpreter();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("forbidden.safetensors");
    let source = format!("budget(0); protect({{'b1','b1'}}); protect('b1'); n=mlp([2 2],'relu'); n.b1[0]=1; wsave(n,'{}');", path.display());
    assert_eq!(run(&mut it, &source).err().unwrap().code, "E-PROTECTED");
    assert!(!path.exists());
    assert_eq!(it.harness.protect, vec!["b1"]);
    assert_eq!(it.harness.used, 0);
}

#[test]
fn inline_configuration_rejects_bad_values_without_spending_passes() {
    for source in [
        "budget(-1);",
        "budget(NaN);",
        "budget(Inf);",
        "budget(0.5);",
        "budget(1e20);",
        "budget(1,2);",
        "dataset('',[1]);",
        "dataset('a','text');",
        "dataset('a');",
        "protect('');",
        "protect({'b1',3});",
        "protect();",
    ] {
        let mut it = interpreter();
        assert!(run(&mut it, source).is_err(), "{source}");
        assert_eq!(it.harness.used, 0);
        assert!(it.harness.protect.is_empty());
    }
}
