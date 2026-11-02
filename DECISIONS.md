# Decisions

<!-- Fill in each section at the step the activity website gives, then commit and push.
     Write each answer after its label. Keep the headings and labels exactly as they are:
     `python -m analysis` reads them. Limit: 150 words for Before results.
     Fill in Recommendations and Design choice and AI use last, once your app is built. -->

## Before results
For each dataset: positive class; which error matters more for the job (missed positive or false alarm) and why.
- readmission: Positive class: readmitted within 30 days of discharge. A missed readmission matters more because a patient the model doesn't flag goes home without extra follow-up and may relapse with worse conditions. A false alarm only costs an unneeded follow-up call or visit, although flagging too many patients could increase costs and stretch hospital staff.
- diabetes: Positive class: has diabetes (doctor diagnosis or HbA1c ≥ 6.5%). A missed positive matters more because that person goes undiagnosed and untreated, with a higher chance of damage to the eyes, kidneys, nerves and heart. A false alarm only costs one extra follow-up blood test, a needle prick and some worry.

## Recommendations
- readmission: <model and threshold, or "none is adequate for this job"> because ...
- diabetes: ...

## Design choice and AI use
A few sentences each. The design choice must be your own idea, not an AI tool's.
- design choice: <one thing you chose to show, or leave out, for your clinical lead> because ...
- AI suggestion I turned down: <what an AI tool suggested, and why you didn't use it, or "no AI tools used">
