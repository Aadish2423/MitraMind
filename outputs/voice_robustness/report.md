# Voice chat with muffled, quiet, noisy and phone audio

12 clean English recordings (Windows text-to-speech, 2 voices, 6 sentences including 2 crisis sentences and 1 'burden' sentence), each degraded 8 ways and transcribed by whisper:openai/whisper-small (language auto-detected, as in the app) with 4 clean-up options. Noisy conditions use 5 different noise draws per clip. WER = share of words wrong (lower is better). 'Crisis caught' = high/urgent sentences that still got high/urgent from the transcript; 'any alert' = they at least got watch or above (an alert for the care team and a helpline in the reply).

| Condition | Clean-up | WER | Crisis caught | Crisis, any alert | Risk level kept | False alarms |
|---|---|---|---|---|---|---|
| clean | none | 0.000 | 4/4 | 4/4 | 6/6 | 0 |
| clean | normalise | 0.000 | 4/4 | 4/4 | 6/6 | 0 |
| clean | highpass,normalise | 0.000 | 4/4 | 4/4 | 6/6 | 0 |
| clean | highpass,denoise,normalise | 0.000 | 4/4 | 4/4 | 6/6 | 0 |
| muffled (cloth over mic, <1.5 kHz) | none | 0.000 | 4/4 | 4/4 | 6/6 | 0 |
| muffled (cloth over mic, <1.5 kHz) | normalise | 0.000 | 4/4 | 4/4 | 6/6 | 0 |
| muffled (cloth over mic, <1.5 kHz) | highpass,normalise | 0.000 | 4/4 | 4/4 | 6/6 | 0 |
| muffled (cloth over mic, <1.5 kHz) | highpass,denoise,normalise | 0.000 | 4/4 | 4/4 | 6/6 | 0 |
| very muffled (<700 Hz) | none | 0.000 | 4/4 | 4/4 | 6/6 | 0 |
| very muffled (<700 Hz) | normalise | 0.000 | 4/4 | 4/4 | 6/6 | 0 |
| very muffled (<700 Hz) | highpass,normalise | 0.000 | 4/4 | 4/4 | 6/6 | 0 |
| very muffled (<700 Hz) | highpass,denoise,normalise | 0.000 | 4/4 | 4/4 | 6/6 | 0 |
| very quiet (-30 dB) | none | 0.000 | 4/4 | 4/4 | 6/6 | 0 |
| very quiet (-30 dB) | normalise | 0.000 | 4/4 | 4/4 | 6/6 | 0 |
| very quiet (-30 dB) | highpass,normalise | 0.000 | 4/4 | 4/4 | 6/6 | 0 |
| very quiet (-30 dB) | highpass,denoise,normalise | 0.000 | 4/4 | 4/4 | 6/6 | 0 |
| noisy room (SNR 10 dB) | none | 0.000 | 20/20 | 20/20 | 30/30 | 0 |
| noisy room (SNR 10 dB) | normalise | 0.000 | 20/20 | 20/20 | 30/30 | 0 |
| noisy room (SNR 10 dB) | highpass,normalise | 0.000 | 20/20 | 20/20 | 30/30 | 0 |
| noisy room (SNR 10 dB) | highpass,denoise,normalise | 0.001 | 20/20 | 20/20 | 30/30 | 0 |
| very noisy (SNR 0 dB) | none | 0.009 | 20/20 | 20/20 | 25/30 | 0 |
| very noisy (SNR 0 dB) | normalise | 0.009 | 20/20 | 20/20 | 25/30 | 0 |
| very noisy (SNR 0 dB) | highpass,normalise | 0.008 | 20/20 | 20/20 | 25/30 | 0 |
| very noisy (SNR 0 dB) | highpass,denoise,normalise | 0.006 | 20/20 | 20/20 | 25/30 | 0 |
| phone call (8 kHz) | none | 0.000 | 4/4 | 4/4 | 6/6 | 0 |
| phone call (8 kHz) | normalise | 0.000 | 4/4 | 4/4 | 6/6 | 0 |
| phone call (8 kHz) | highpass,normalise | 0.000 | 4/4 | 4/4 | 6/6 | 0 |
| phone call (8 kHz) | highpass,denoise,normalise | 0.000 | 4/4 | 4/4 | 6/6 | 0 |
| muffled + quiet + noisy | none | 0.135 | 16/20 | 18/20 | 16/30 | 0 |
| muffled + quiet + noisy | normalise | 0.120 | 18/20 | 20/20 | 18/30 | 0 |
| muffled + quiet + noisy | highpass,normalise | 0.124 | 17/20 | 20/20 | 17/30 | 0 |
| muffled + quiet + noisy | highpass,denoise,normalise | 0.128 | 16/20 | 19/20 | 16/30 | 0 |

Mean WER over all conditions: none 0.036, normalise 0.032, highpass,normalise 0.033, highpass,denoise,normalise 0.034

Worst transcripts (no clean-up):

- muffled + quiet + noisy, work_stress__zira (WER 0.538): "This voice has been very stressful in my back since giving me my deadline."
- muffled + quiet + noisy, work_stress__zira (WER 0.462): "This voice has been very stressful in my past, it's giving me more deadline."
- muffled + quiet + noisy, work_stress__zira (WER 0.385): "This voice has been very stressful in my past, it's giving me more deadlines."
- muffled + quiet + noisy, work_stress__zira (WER 0.385): "My voice has been very stressful and my back keeps killing me more deadline."
- muffled + quiet + noisy, anxious_exams__zira (WER 0.333): "I have been hearing lay answers about my attempts and I cannot breathe at night."
- muffled + quiet + noisy, crisis_die__zira (WER 0.333): "I want you back. I cannot leave this anymore."
- muffled + quiet + noisy, work_stress__david (WER 0.308): "This work has been very stressful and my boss is getting me more deadline."
- muffled + quiet + noisy, coping_walk__zira (WER 0.286): "I went for a walk again on a breathing exercise helping me feel calmer."
