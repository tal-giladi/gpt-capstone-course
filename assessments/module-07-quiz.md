# Module 7 quiz

Nine scenario questions on running the main training job on rented hardware: choosing where and how
to rent an 8×H100 node for a few hours, the pre-flight checks that protect the money, reading a
long run's curves, deciding when a loss spike needs a rollback, recovering from slowdowns, hangs and
dead nodes, keeping the cost on plan, and finishing cleanly with an exported model and a reconciled
budget. Pass mark: 70% (7 of 9).

Before you start, you should have finished:

- [07.1 · Pre-flight checklist and launch](../lessons/module-07/lesson-01.md)
- [07.2 · Babysitting a long run](../lessons/module-07/lesson-02.md)
- [07.3 · Finishing the run and shutting everything down](../lessons/module-07/lesson-03.md)

The practical result of this module is your base model in `hf/my-gpt-350m-base/`, the final
checkpoint in your bucket, and `reports/main-run-log.md` with every incident and the real cost.
The quiz checks the decisions behind them.
