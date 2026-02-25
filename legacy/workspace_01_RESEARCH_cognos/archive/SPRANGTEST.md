Sprängtest (thought experiment) för CognOS

Syfte

Detta sprängtest avgör om CognOS är värt att driva vidare genom att testa om
ett decision layer ger mätbara effekter ovanpå CUDA/Tensor-stacken.

Nyckelprincip

CognOS förändrar inte beräkningen, utan hur beräkning används.
Effekten ska därför synas i beslutskvalitet, kostnad, tillit och governance.

Baseline vs CognOS

Baseline:
GPU/CUDA -> Tensor frameworks -> Model -> Application -> Human decision

CognOS:
GPU/CUDA -> Tensor frameworks -> Models -> CognOS Runtime -> Applications/Humans

Struktur: 3 × 2-veckors bets + 1 × 6-veckors bet

Varje cykel har sin egen circuit breaker. Om en cykel misslyckas,
stoppa och omvärdera innan du går vidare. Detta följer principerna i
[ARBETSSATT.md](../FORSKNING/ARBETSSATT.md): fixed appetite, circuit breaker, en sak i taget.

Cykel 1 — Confidence score (2 veckor)

Fråga: Ger en enkel confidence-score C = p(x) × (1 - Ue) värde som filter?

- Bygg en Python-funktion: model output + MC Dropout variance → C ∈ [0,1]
- Testa på 100+ datapunkter (syntetisk eller offentlig dataset)
- Sätt en threshold τ och mät: filterar C bort fel som baseline missar?

GO om: confidence-score minskar andelen överkonfidenta fel med >30%.
NO-GO om: ingen mätbar skillnad mot baseline.

Biprodukt: tekniskt blogginlägg om epistemisk osäkerhet i inference.

Cykel 2 — Model routing (2 veckor)

Fråga: Sparar dynamisk modellrouting pengar vid bibehållen kvalitet?

- 3 modeller: small, medium, large (eller motsvarande)
- Routing-logik: small först → om C < τ → eskalera till medium → large
- Jämför: statisk large-modell vs routing. Mät kostnad och kvalitet.

GO om: 20–40% lägre inferenskostnad vid samma eller bättre kvalitet.
NO-GO om: routing ger försumbar besparing eller sämre kvalitet.

Biprodukt: benchmark-dataset + kostnadsanalys.

Cykel 3 — Sjukvårdstriage (2 veckor)

Fråga: Fungerar confidence + routing i en riktig domän?

- Använd offentlig triagedata (eller syntetisk sjukvårdsdata)
- CognOS-pipeline: patientdata → modell → confidence → routing → rekommendation
- Mät: precision, recall, andel eskalerade (human-in-loop) fall

GO om: CognOS-triage visar bättre precision eller identifierar fler osäkra fall
korrekt jämfört med en statisk modell.
NO-GO om: ingen mätbar förbättring i triagekvalitet.

Biprodukt: case study "CognOS i sjukvårdstriage".

Cykel 4 — MVP med audit trail (6 veckor)

Förutsättning: minst 2 av 3 tidigare cykler = GO.

- Integrera confidence engine + routing + audit trail i en sammanhållen runtime
- Sjukvårdstriage som vertikal
- Output per beslut: rekommendation, C-score, vald modell, motivering, spårbar logg
- 100% audit coverage för alla beslut i pilotflödet

GO om: fungerande MVP som kan demonstreras för en potentiell kund eller partner.
NO-GO om: systemet inte håller ihop som helhet.

Biprodukt: teknisk artikel redo för submission.

Beslutsmatris (total)

GO (hela projektet):
- Minst 2 av 3 tidiga cykler uppfyllda
- Cykel 4 levererar en fungerande demo
- Minst 1 tydlig KPI-förbättring som kan säljas

NO-GO:
- Färre än 2 av 3 tidiga cykler uppfyllda
- Inga säljbara KPI-förbättringar
- I det fallet: omformulera scope eller avsluta

Nästa steg efter GO

- Vertikal pilot med riktig kund/partner
- Publicera teknisk artikel
- Starta kunddialog med mätbar ROI
