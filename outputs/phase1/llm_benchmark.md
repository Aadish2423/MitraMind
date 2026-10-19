# LLM benchmark: State Card narrative (structured output)

First-attempt results per demo patient (no retry). `pass` = schema-valid and no guardrail issue.

| Model | Schema-valid | Guardrail pass | Mean latency (s) | Mean summary words |
|---|---|---|---|---|
| ollama:qwen3.5(think=True) | 1/4 | 1/4 | 241.1 | 62 |
| ollama:qwen3.5:4b(think=True) | 2/4 | 2/4 | 119.5 | 54 |
| ollama:mistral | 4/4 | 2/4 | 20.7 | 87 |
| ollama:gemma4:e4b(think=False) | 4/4 | 4/4 | 10.2 | 47 |
| ollama:qwen3.5:4b(think=False) | 4/4 | 2/4 | 12.1 | 76 |
| ollama:qwen3.5(think=False) | 4/4 | 4/4 | 20.3 | 56 |

## Per patient

| Model | Patient | s | Valid | Pass | Issues |
|---|---|---|---|---|---|
| ollama:qwen3.5(think=True) | demo_caregiver_03 | 255.1 | False | False | schema: Invalid JSON: EOF while parsing a value at line 1 column 0 |
| ollama:qwen3.5(think=True) | demo_crisis_04 | 269.2 | False | False | schema: Invalid JSON: EOF while parsing a value at line 1 column 0 |
| ollama:qwen3.5(think=True) | demo_professional_02 | 216.2 | True | True | - |
| ollama:qwen3.5(think=True) | demo_student_01 | 224.0 | False | False | schema: Invalid JSON: EOF while parsing a value at line 1 column 0 |
| ollama:qwen3.5:4b(think=True) | demo_caregiver_03 | 114.7 | True | True | - |
| ollama:qwen3.5:4b(think=True) | demo_crisis_04 | 131.0 | True | True | - |
| ollama:qwen3.5:4b(think=True) | demo_professional_02 | 117.2 | False | False | schema: Invalid JSON: EOF while parsing a value at line 1 column 0 |
| ollama:qwen3.5:4b(think=True) | demo_student_01 | 115.1 | False | False | schema: Invalid JSON: EOF while parsing a string at line 40 column 3 |
| ollama:mistral | demo_caregiver_03 | 19.4 | True | False | diagnostic language ('mild depression'); report the score and severity band (e.g. 'PHQ-9 17, moderately severe range') and describe reported experiences instead |
| ollama:mistral | demo_crisis_04 | 16.3 | True | False | diagnostic language ('level of depression'); report the score and severity band (e.g. 'PHQ-9 17, moderately severe range') and describe reported experiences instead |
| ollama:mistral | demo_professional_02 | 22.0 | True | True | - |
| ollama:mistral | demo_student_01 | 25.2 | True | True | - |
| ollama:gemma4:e4b(think=False) | demo_caregiver_03 | 9.0 | True | True | - |
| ollama:gemma4:e4b(think=False) | demo_crisis_04 | 8.3 | True | True | - |
| ollama:gemma4:e4b(think=False) | demo_professional_02 | 11.7 | True | True | - |
| ollama:gemma4:e4b(think=False) | demo_student_01 | 11.8 | True | True | - |
| ollama:qwen3.5:4b(think=False) | demo_caregiver_03 | 11.0 | True | True | - |
| ollama:qwen3.5:4b(think=False) | demo_crisis_04 | 11.4 | True | False | '0 of 3 check-ins' does not match the data (3 check-ins; per-domain counts [1, 3]) |
| ollama:qwen3.5:4b(think=False) | demo_professional_02 | 13.8 | True | False | '2 of 7 check-ins' does not match the data (7 check-ins; per-domain counts [1, 7]) |
| ollama:qwen3.5:4b(think=False) | demo_student_01 | 12.0 | True | True | - |
| ollama:qwen3.5(think=False) | demo_caregiver_03 | 15.3 | True | True | - |
| ollama:qwen3.5(think=False) | demo_crisis_04 | 18.9 | True | True | - |
| ollama:qwen3.5(think=False) | demo_professional_02 | 25.9 | True | True | - |
| ollama:qwen3.5(think=False) | demo_student_01 | 21.2 | True | True | - |

## Summaries

**ollama:qwen3.5(think=True) / demo_caregiver_03:** 

**ollama:qwen3.5(think=True) / demo_crisis_04:** 

**ollama:qwen3.5(think=True) / demo_professional_02:** Over 20 days with 7 check-ins, severity worsened. PHQ-9 total increased from 8 to 17 (+9), and GAD-7 total increased from 9 to 15 (+6). Patient reported fatigue (p02), insomnia (p02), irritability (p05), and conflict at home involving work (p01) and economic stressors (p05). Substance coping (p04) and breathing exercises (p06) were noted as not helpful. No effective coping strategies were identified.

**ollama:qwen3.5(think=True) / demo_student_01:** 

**ollama:qwen3.5:4b(think=True) / demo_caregiver_03:** Patient reported mild PHQ-9 (5) and minimal GAD-7 (4) with stable scores over 2 weeks. Stressors included family (sister/mother) and physical health concerns (hospital/doctor). Coping strategies like routine selfcare and mindfulness were marked helpful in c02 and c04. Sleep was not reported in check-ins.

**ollama:qwen3.5:4b(think=True) / demo_crisis_04:** Patient reported a PHQ-9 total of 18 on 2026-10-01 with stable severity over the period. Three check-ins occurred between 2026-10-02 and 2026-10-08 (u01, u02, u03). Stressors included relationships (breakup) in u01 and physical health (pain) in u02, while self-harm thoughts were reported in PHQ-9 domains during the week of 2026-10-05 (u03). Patient mentioned social support and professional help as coping strategies in u02.

**ollama:qwen3.5:4b(think=True) / demo_professional_02:** 

**ollama:qwen3.5:4b(think=True) / demo_student_01:** {
  "clinician_summary": "Patient reported PHQ-9 improvement from 14 to 9 (improved by -5) and GAD-7 improvement from 12 to 8 (improved by -4). Sleep issues were reported in s01 and s06, alongside exam stressors in 6 of 8 check-ins. Helpful coping included breathing exercises (s03) and social suppor

**ollama:mistral / demo_caregiver_03:** Patient reported mild depression (PHQ-9: 5 -> 4) and minimal anxiety (GAD-7: 4) over the past two weeks, with stable scores. The patient mentioned trouble with sleep (tiredness) and nervousness. Family and physical health were reported as stressors, with the patient discussing these issues in three out of four check-ins. The patient reported routine self-care and mindfulness as helpful coping strategies. The patient also shared that they practiced yoga in the morning and felt calm. Discussion points may include the patient's current stressors, their sleep patterns, and further exploration of their coping strategies.

**ollama:mistral / demo_crisis_04:** The patient reported a moderately severe level of depression (PHQ-9 total: 18, severity: moderately severe) during the past week, with no change from the baseline. They mentioned trouble with relationships and physical health as stressors, and reported seeking social support and professional help as coping strategies. The patient also reported thoughts of self-harm in the most recent check-in.

**ollama:mistral / demo_professional_02:** The patient's PHQ-9 and GAD-7 scores have worsened over the past three weeks, with PHQ-9 increasing by 9 points (from mild to moderately severe) and GAD-7 increasing by 6 points (from mild to severe). The patient reported trouble sleeping, fatigue, loss of interest, low mood, and irritability. Stressors mentioned include work, relationships, conflict at home, alcohol/substances, and economic issues. The patient reported that alcohol and breathing strategies have not been helpful in coping. Discussion points may include the patient's current mood, sleep problems, and stressors, as well as the impact of alcohol on their mental health.

**ollama:mistral / demo_student_01:** The patient's PHQ-9 and GAD-7 scores have shown improvement over the past three weeks, with a decrease of 5 points in PHQ-9 and 4 points in GAD-7. The patient reported trouble sleeping in 2 out of 8 check-ins, and mentioned guilt/worthlessness, tiredness, and poor concentration once each. Uncontrollable worry, panic/bodily anxiety, and nervousness were reported once each in GAD-7 domains. The patient's primary stressors were studies/exams, mentioned in 6 check-ins, and family, mentioned in 2 check-ins. Breathing exercises and social support were reported as helpful coping strategies. Discussion points may include the patient's sleep issues, ongoing stressors related to studies/exams, and the

**ollama:gemma4:e4b(think=False) / demo_caregiver_03:** Over the last 2 weeks, PHQ-9 score decreased from 5 to 4, and GAD-7 remained stable. The patient reported stressors related to family (3 check-ins) and physical health (3 check-ins). Helpful coping strategies included routine self-care and mindfulness. Discussion points include family and health stressors.

**ollama:gemma4:e4b(think=False) / demo_crisis_04:** The patient's PHQ-9 score was 18 at baseline. Across 3 check-ins, the patient mentioned stressors related to relationships and physical health. The patient reported utilizing social support and professional help in 1 check-in. Discussion points include the impact of relationship issues and physical health concerns.

**ollama:gemma4:e4b(think=False) / demo_professional_02:** PHQ-9 score worsened from 8 to 17, and GAD-7 score worsened from 9 to 15 over 3 weeks. The patient reported experiencing tiredness, sleep problems, and low mood in check-ins. Stressors included work, relationships, and economic concerns. Unhelpful coping strategies mentioned include substance use and breathing exercises.

**ollama:gemma4:e4b(think=False) / demo_student_01:** PHQ-9 score improved from 14 to 9, and GAD-7 improved from 12 to 8 over 3 weeks. The patient reported 2 instances of sleep problems and 1 instance of guilt/worthlessness. Stressors included school exams (6 check-ins) and family issues (2 check-ins). Breathing and social support were reported as helpful coping strategies.

**ollama:qwen3.5:4b(think=False) / demo_caregiver_03:** Over the past two weeks, the patient completed four check-ins with a PHQ-9 score of 5 (mild) at the start and 4 (minimal) at the end, showing a stable trend. GAD-7 scores were minimal (4). The patient reported tiredness in one check-in and nervousness in another. Primary stressors involved family issues regarding their sister and mother, as well as physical health concerns including hospital visits and doctor reports. Helpful coping strategies included routine self-care and mindfulness. No unhelpful coping or sleep disturbances were noted.

**ollama:qwen3.5:4b(think=False) / demo_crisis_04:** Patient reported baseline PHQ-9 severity of 18 (moderately severe) with no change. They mentioned trouble sleeping in 0 of 3 check-ins and discussed relationships and physical health pain in 1 of 3 check-ins each. Patient noted thoughts of self-harm in 1 of 3 check-ins. Coping strategies included social support and professional help, reported as helpful in 1 of 2 check-ins. No unhelpful coping was reported.

**ollama:qwen3.5:4b(think=False) / demo_professional_02:** Patient reported worsening symptoms over the past three weeks, with PHQ-9 increasing from 8 to 17 and GAD-7 from 9 to 15. They mentioned trouble sleeping in 2 of 7 check-ins, including insomnia and small hours. Reported stressors included work pressure, conflict at home, financial strain, and substance use. Patient reported using alcohol and breathing exercises as coping strategies, both marked as unhelpful or not working. No helpful coping strategies were identified.

**ollama:qwen3.5:4b(think=False) / demo_student_01:** Patient reported improvement in mood and anxiety over the past three weeks, with PHQ-9 scores decreasing from 14 to 9 (moderate to mild) and GAD-7 scores from 12 to 8 (moderate to mild). Sleep difficulties persisted across multiple check-ins, including insomnia and small hours. Primary stressors were academic exams and family comparisons; the patient mentioned feeling like a failure during one check-in. Helpful coping strategies included breathing exercises and social support, while creative music and physical activity were noted as attempted but unverified approaches.

**ollama:qwen3.5(think=False) / demo_caregiver_03:** Over the past two weeks, PHQ-9 scores remained stable (5 to 4) and GAD-7 was minimal (4). The patient reported tiredness in one check-in and nervousness in another. Stressors included family matters and physical health concerns. Helpful coping strategies included routine self-care and mindfulness.

**ollama:qwen3.5(think=False) / demo_crisis_04:** The patient completed three check-ins over six days. PHQ-9 severity was moderately severe (18) at baseline with no change reported. Reported domains included low mood, thoughts of self-harm, relationships, and physical health. The patient mentioned social support and professional help as coping strategies.

**ollama:qwen3.5(think=False) / demo_professional_02:** Over the past 20 days, PHQ-9 scores worsened from 8 to 17 and GAD-7 scores worsened from 9 to 15. The patient reported fatigue, sleep problems, loss of interest, low mood, irritability, and anhedonia across seven check-ins. Stressors included work pressure, relationship conflict, domestic fights, economic concerns, and substance use. No coping strategies were reported as helpful; alcohol/substance use and breathing exercises were noted as unhelpful.

**ollama:qwen3.5(think=False) / demo_student_01:** Over the past three weeks, PHQ-9 scores improved from 14 to 9 and GAD-7 scores decreased from 12 to 8. The patient reported sleep difficulties in 2 of 8 check-ins and cited studies/exams as a primary stressor in 6 entries. Helpful coping strategies included breathing exercises and social support, while creative music and physical activity were mentioned but not marked as helpful. Family dynamics were noted in 2 check-ins.
