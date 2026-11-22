TODOS

- fix attention
- natrenovat primo jednoho chytace jak si vede 
- jak funguje jeden chytac? 
- evader APF vic agilni. 

- lr ,ent coeff na konci snizovat (to znamena konst a pak linear decay or something)
- nahodny restarty, delka epizody
- koeff pro penalizaci cmd, postupne zvysovat
- penazilace velikosti vah v rewardu primo
- velikost site pro chytani jako curriculum

## The plan

All runs use the MLP with the settings of `04_new_base`, seeds 0 and 1.

1. **Eval the basic APF.** Fly the two `04_new_base` policies against it, 4096 episodes. This needs no new code.
2. **Test the agile variants, eval only.** Use the same policies, push factor 1, 2 and 4, each with and without the limits. You pick one from the table.
3. **Add the code.** One pilot `apf_agile`, and the presets `apf_1v1`, `apf_3v1`, `apf_agile_1v1` and `apf_agile_3v1`. Update the chase spec.
4. **Train the swarm.** `apf_3v1` and `apf_agile_3v1`.
5. **Train one pursuer.** `paths_1v1`, `apf_1v1` and `apf_agile_1v1`, with the same steps.
6. **Make one table.** Each policy against paths, APF and agile APF. The gain of the swarm is the 3v1 catch minus the 1v1 catch.







### Other approaches

| approach | from | new code |
|---|---|---|
| lr and `ent_coef` constant, then a linear fall at the end | line 1 above | yes |
| each env starts its first episode at a random time | line 2 above | yes, the trainer has none now |
| a `cmd` cost that grows over the training | line 3 above | yes |
| a cost on the size of the network weights, in the loss | line 4 above | yes, the optimizer has none now |
| a floor for the std, for example 0.3 | the log, Open | yes |
| train in `paths_3v1_netpass`, eval in `paths_3v1` | the log | no |
| `paths_3v1_body`, the body catch | the log | no |
| the best MLP setup with `net=attention` | you | no |
