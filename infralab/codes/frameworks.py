"""Normative built-in.

- custom: passthrough (usa i target manuali dei parametri; default storico).
- EN1990: target β di riferimento per classe di conseguenza (EN 1990:2002,
  Tab. B2, periodo 50 anni). CC1/CC2/CC3 = 3.3/3.8/4.3, β_SLS = 1.5.
- fib_MC2010: Model Code 2010 (§3.3), target β 50 anni allineati a EN1990 con
  incremento per rottura fragile (§7.3) — CC1/CC2/CC3 = 3.1/3.8/4.4.
- AASHTO_LRFD: target di componente β = 3.5 su vita utile 75 anni (AASHTO LRFD
  Bridge Design Specifications, calibrazione Nowak NCHRP 368).

I valori sono documentati nelle sorgenti citate; sostituibili senza toccare il
motore. Servono a valutare l'effetto della normativa sulle decisioni.
"""
from infralab.codes.base import CodeFramework
from infralab.codes.registry import register_code_framework

register_code_framework(CodeFramework(
    name="custom",
    display_name="Personalizzata (parametri manuali)",
    source="—",
))

register_code_framework(CodeFramework(
    name="EN1990",
    display_name="Eurocodice EN 1990",
    source="EN 1990:2002, Tab. B2 (periodo di riferimento 50 anni)",
    beta_reference_by_cc={"CC1": 3.3, "CC2": 3.8, "CC3": 4.3},
    reference_period_years=50,
    sls_beta_reference=1.5,
    brittle_beta_increment=0.0,
))

register_code_framework(CodeFramework(
    name="fib_MC2010",
    display_name="fib Model Code 2010",
    source="fib Model Code 2010, §3.3 / §7.3",
    beta_reference_by_cc={"CC1": 3.1, "CC2": 3.8, "CC3": 4.4},
    reference_period_years=50,
    sls_beta_reference=1.5,
    brittle_beta_increment=0.5,
))

register_code_framework(CodeFramework(
    name="AASHTO_LRFD",
    display_name="AASHTO LRFD (US)",
    source="AASHTO LRFD Bridge Design Spec.; Nowak NCHRP 368 (vita 75 anni)",
    beta_reference_by_cc={"CC1": 3.5, "CC2": 3.5, "CC3": 3.75},
    reference_period_years=75,
    sls_beta_reference=1.0,
    brittle_beta_increment=0.25,
))

# --- Tier per STRUTTURE ESISTENTI (valutazione, non nuova costruzione) -------
# NEN 8700:2011 (+A1:2020), "Assessment of existing structures in case of
# reconstruction and disapproval": target β distinti dai valori di progetto
# EN 1990, riferiti alla vita residua (minimo 15 anni; 30 raccomandati).
# Valori VERIFICATI (2026-09-17) sulla riproduzione della tabella dei minimi
# NEN 8700 in de Vries, Steenbergen, Maljaars, "Annual reliability requirements
# for bridges and viaducts", HERON 68 (2023) No. 2, Tab. 3, colonna "wn" (vento
# non dominante), periodo di riferimento 15 anni: disapprovazione CC1b/CC2/CC3 =
# 1.8/2.5/3.3; rinnovo (renovation) 2.8/3.3/3.8. CC1 qui = CC1b. Confronto
# ROK/RBK (RWS 2013): minimo 2.5 @15 a, uso 3.3 @30 a, nuovo 4.3 @100 a.
# Riferimenti: ISO 13822:2010; ISO 2394:2015; fib Bulletin 80 (2016).
register_code_framework(CodeFramework(
    name="NEN8700_disapproval",
    display_name="NEN 8700 - strutture esistenti, livello di disapprovazione (15 anni)",
    source="NEN 8700:2011 (disapproval, wn) come riprodotta in HERON 68(2) 2023 Tab. 3; ISO 13822; fib Bulletin 80",
    beta_reference_by_cc={"CC1": 1.8, "CC2": 2.5, "CC3": 3.3},
    reference_period_years=15,
    sls_beta_reference=1.5,
    brittle_beta_increment=0.0,
))
register_code_framework(CodeFramework(
    name="NEN8700_reconstruction",
    display_name="NEN 8700 - strutture esistenti, livello di rinnovo (15 anni)",
    source="NEN 8700:2011 (renovation, wn) come riprodotta in HERON 68(2) 2023 Tab. 3; ISO 13822; fib Bulletin 80",
    beta_reference_by_cc={"CC1": 2.8, "CC2": 3.3, "CC3": 3.8},
    reference_period_years=15,
    sls_beta_reference=1.5,
    brittle_beta_increment=0.0,
))
