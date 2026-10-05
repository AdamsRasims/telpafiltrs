# TelpaFiltrs N2

Ādams Rasims, 231RDB279. Nepieciešams Python 3.9 vai jaunāks.
Papildu bibliotēkas nav jāinstalē. Atarhivē ZIP un tā mapē izpildi:

```text
python n2.py
```

Programma nolasa trīs CSV failus un terminālī JSON formā izvada visus
atskaites aprēķinus: pazīmes, min/max, stereotipu skaitu, k=2,3,4 rezultātus,
klašu centrus, katra lietotāja klasi, atšķirīgos piekārtojumus, piecu jaunu
lietotāju klasifikāciju, EMA un histerēzi. Skaitļi izvadē ir pilnā precizitātē;
PDF tie noapaļoti. Komanda nemaina ievaddatus.

Faili:
- `n2.py`: ģenerators un visi aprēķini, tikai Python standarta bibliotēka.
- `zurnals.csv`: apmācības dati, 36 lietotāji, 4 sesijas, 12 mēģinājumi sesijā.
- `testa_zurnals.csv`: 5 atsevišķi testa lietotāji, katram 4 sesijas.
- `laika_zurnals.csv`: U01 turpinājums S5-S10 EMA un histerēzes demonstrācijai.

Visi dati ir sintētiski. Ģenerēšanas seed: 231279, 231280 un 231281.
CSV var identiski atjaunot ar `python n2.py --generate`.

CSV shēma: user_id, session_id, timestamp (sintētisks Unix laiks sekundēs),
attempt_id, event_type, value, or_count, not_count, max_depth.
`rule_submit` glabā mēģinājuma aktīvo laiku un loģisko konstrukciju skaitu;
pārējie notikumi ir syntax_error, semantic_error, saved_rule_reused un to
value=1. Kļūdu skaits attiecas uz mēģinājumiem, nevis tikai veiksmīgām izpildēm.
Vienā mēģinājumā katrs notikuma tips parādās ne vairāk kā vienu reizi.

Modeli un min/max iegūst tikai no zurnals.csv. Testa un turpinājuma dati
neietekmē apmācību. k-vidējo: 20 atšķirīgas inicializācijas katram k,
izvēle pēc mazākās kvadrātisko attālumu summas; siluets ar Eiklīda attālumu.
Vienādu attālumu gadījumā izvēlas pirmo centru sarakstā. Tukšs klasteris
izraisa jaunu inicializāciju. Konstanta pazīme normalizējas uz nulli.
Izvēlētais k=3 ir apzināts dialoga projektēšanas lēmums, nevis silueta maksimums.
Sākotnējais U01 modelis ir pirmo četru sesiju pazīmju vektors; pēc tam
visas piecas pazīmes atjaunina ar EMA 0,3 un klasei piemēro δ=0,1, N=2.
Sākotnējai klasificēšanai pārbauda vismaz četras sesijas un 48 mēģinājumus.
Aprēķinātā klase ir modeļa novērtējums; lietotnes režīma maiņai paredzēts
atsevišķs lietotāja apstiprinājums un iespēja atgriezt iepriekšējo režīmu.
N2 skripts aprēķina modeli; interfeisa apstiprinājumu darbības ir dialoga
profila prasības, kas šajā nodevumā nav jāprogrammē.

N2 sintētiskie CSV tiek saglabāti atkārtojamībai. Projektējamā lietotnē
žurnālu paredzēts glabāt servera datubāzē 90 dienas, modeli līdz profila
dzēšanai vai 90 dienu neaktivitātei. Šī politika nav CSV automātiskas dzēšanas
funkcija. Atskaite papildināta pēc pasniedzēja N1 atgriezeniskās saites.

Pamatmetodes pielāgotas no kursa L4_lietotaju_modelis.py. MI asistents
OpenAI Codex izmantots koda, sintētisko datu ģeneratora un atskaites izstrādei
un rezultātu pārbaudei. Rezultāti iegūti, izpildot šo programmu.
