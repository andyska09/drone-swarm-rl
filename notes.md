TODOS

- fix attention
- natrenovat primo jednoho chytace jak si vede 
- jak funguje jeden chytac? 
- evader APF vic agilni. 


- resolve the growing entropy in 3v1 training and why it failed to learn to catch the faster APF
- finalize the attention design 
- slower pursuer, edit the drone physics. APF keeps its limits from Nemecek. -> this way we train the 1v1 and 3v1 case 

## Tips and tricks to try when something isnt training

- lr ,ent coeff na konci snizovat (to znamena konst a pak linear decay or something)
- nahodny restarty, delka epizody
- koeff pro penalizaci cmd, postupne zvysovat
- penazilace velikosti vah v rewardu primo
- velikost site pro chytani jako curriculum

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
