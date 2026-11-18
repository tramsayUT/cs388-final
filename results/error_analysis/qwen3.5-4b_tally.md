# Error-analysis tally: qwen3.5-4b

## Categories by group

| category | sa_high_tp_low | tp_high_sa_low | bias_flip | Total |
|---|---|---|---|---|
| correct | 9 | 3 | 3 | 15 |
| key_error | 1 | 0 | 0 | 1 |
| knowledge_gap | 0 | 5 | 3 | 8 |
| near_miss | 0 | 2 | 1 | 3 |
| artifact | 0 | 0 | 1 | 1 |
| other | 0 | 0 | 2 | 2 |

## Note tags by group (can overlap)

| tag | sa_high_tp_low | tp_high_sa_low | bias_flip | Total |
|---|---|---|---|---|
| Negation question | 6 | 1 | 2 | 9 |
| Garbled dataset text | 4 | 2 | 1 | 7 |
| Letter bias | 0 | 0 | 2 | 2 |
| Flawed reasoning, right answer | 1 | 0 | 0 | 1 |
| Key questionable | 1 | 0 | 1 | 2 |
| Disagreeing sample was right | 0 | 3 | 2 | 5 |
| Two plausible options | 0 | 1 | 0 | 1 |

## Raw token-probability answer correct (out of 10)

| measure | sa_high_tp_low | tp_high_sa_low | bias_flip | Total |
|---|---|---|---|---|
| Raw answer correct | 9 | 3 | 3 | 15 |

## Bias flips: outcome by raw letter

| outcome | A | B | C | D | Total |
|---|---|---|---|---|---|
| fixed | 0 | 3 | 1 | 0 | 4 |
| broke | 0 | 2 | 0 | 1 | 3 |
| both wrong | 2 | 0 | 0 | 1 | 3 |

## Per-row tags (check for missed keywords)

| id | group | gold | tp_raw_answer | tp_corr_answer | category | tags | flip |
|---|---|---|---|---|---|---|---|
| 1cd8474b-6560-484d-b733-9ba499981029 | sa_high_tp_low | D | C | C | key_error | Negation question; Key questionable |  |
| fa304b14-4fa8-415b-8fba-3e670529cc8a | sa_high_tp_low | A | A | D | correct |  |  |
| 5dd20354-df6d-4a05-9895-d742e7852803 | sa_high_tp_low | B | B | B | correct | Negation question |  |
| 64400a13-aab9-46fd-8609-ac725bb8e72e | sa_high_tp_low | C | C | C | correct | Negation question; Flawed reasoning, right answer |  |
| 60b9f4b0-f993-4236-9378-3c87762aefff | sa_high_tp_low | B | B | B | correct | Negation question |  |
| 5444cae2-aeab-4a04-8501-e91fe8ce794a | sa_high_tp_low | A | A | A | correct | Garbled dataset text |  |
| f4affb3a-0fce-423d-aaac-027d30259334 | sa_high_tp_low | D | D | D | correct | Garbled dataset text |  |
| aa23e05a-7e73-494b-bc8d-6331b060a964 | sa_high_tp_low | D | D | D | correct | Negation question; Garbled dataset text |  |
| 95deaff5-dc0b-42a7-b257-573f3de4c88d | sa_high_tp_low | D | D | A | correct | Garbled dataset text |  |
| bf0eabec-95a0-46e1-a9b7-dae6c29dadde | sa_high_tp_low | B | B | B | correct | Negation question |  |
| f955fff3-f4ec-41be-a119-a43e2024120e | tp_high_sa_low | C | B | B | knowledge_gap | Negation question; Disagreeing sample was right |  |
| 12a8e0d0-21d1-4edf-905d-5e9c415b1a80 | tp_high_sa_low | C | B | B | knowledge_gap | Disagreeing sample was right |  |
| 5d0bb1e6-fa95-47e6-811e-abeb5a025ce6 | tp_high_sa_low | C | C | A | correct | Garbled dataset text |  |
| 66092ccf-9a06-466a-a08f-8964f3091a9c | tp_high_sa_low | A | B | B | knowledge_gap |  |  |
| 638938c5-4c5d-4ce0-8742-150395cce724 | tp_high_sa_low | B | B | B | correct |  |  |
| dfb6dc4a-a4ac-4270-8bf2-807e6efead3a | tp_high_sa_low | B | B | B | correct | Two plausible options |  |
| 2166e021-8b96-47ad-9e4d-fa55a623938b | tp_high_sa_low | A | C | C | knowledge_gap |  |  |
| 17b84a1e-02fc-4250-a93a-194093782811 | tp_high_sa_low | A | C | C | near_miss | Garbled dataset text |  |
| 8d2e7fe2-3354-44c2-8cdf-55ff23caaebb | tp_high_sa_low | A | B | A | near_miss |  |  |
| 4d8af4ad-1f62-4bb8-a485-439992b3af49 | tp_high_sa_low | A | D | D | knowledge_gap | Disagreeing sample was right |  |
| 89e0b62e-fa1d-4964-8870-d57cb8b871f4 | bias_flip | B | B | A | correct |  | broke |
| 80ca7074-85db-4ead-9b69-23c32b811d94 | bias_flip | D | D | A | correct | Key questionable | broke |
| 98200364-a113-4848-9149-fdd6106442a5 | bias_flip | D | B | D | near_miss |  | fixed |
| 3e2e85ff-abd6-4b35-bab8-b5fe0b78bdd9 | bias_flip | B | B | D | correct |  | broke |
| f184a533-98b4-43ad-b1f5-70edc6704d9e | bias_flip | A | D | B | knowledge_gap |  | both wrong |
| 79e1618e-0e20-40ba-b104-8397f32985cc | bias_flip | B | A | D | artifact | Garbled dataset text; Disagreeing sample was right | both wrong |
| 3ca099a7-086a-411e-917e-295561ebb22b | bias_flip | D | B | D | other | Letter bias | fixed |
| aceab90d-6cc0-4cac-82d4-8fcda05ea933 | bias_flip | D | C | D | knowledge_gap | Disagreeing sample was right | fixed |
| 7cacd6ea-ee7d-481d-bc66-467c47dc315c | bias_flip | B | A | C | knowledge_gap | Negation question | both wrong |
| 8692345e-a600-4355-a964-bb514b16d977 | bias_flip | A | B | A | other | Negation question; Letter bias | fixed |
