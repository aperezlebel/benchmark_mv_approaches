"""Build prediction tasks for the NHIS database."""

import pandas as pd

from database import TB
from prediction import PredictionTask


df_with_MV = TB.encoded_dataframes['20000']
df_imputed = pd.read_csv('imputed/TB_20000_imputed_rounded_Iterative.csv',
                         sep=';', index_col=0)


# Task 1: Death prediction
to_predict = "Décès"
to_drop_1 = [
    "Date de décès (à l'hôpital après sortie de réanimation)",
    "Cause du décès_A survécu",
    "Cause du décès_Autre (précisez ci-dessous)",
    "Cause du décès_Choc hémorragique",
    "Cause du décès_Choc septique",
    "Cause du décès_Défaillance multi-viscérale",
    "Cause du décès_LATA",
    "Cause du décès_Mort encéphalique",
    "Cause du décès_Trauma cranien",
    "Cause du décès_z MISSING_VALUE",
    "Transfert secondaire, pourquoi ?_Pas de transfert",
    "Transfert secondaire, pourquoi ?_Plateau technique insuffisant",
    "Transfert secondaire, pourquoi ?_Rapprochement familial",
    "Transfert secondaire, pourquoi ?_z MISSING_VALUE",
    "Sortie_Autre réanimation",
    "Sortie_Centre de rééducation",
    "Sortie_Domicile",
    "Sortie_Service hospitalier",
    "Sortie_z MISSING_VALUE",
    "Glasgow de sortie",
    "Nombre de jours à l'hôpital",
    "Durée de séjour en réa- si date de sortie connue, durée de séjour = (date sortie - date d entrée)- si date de sortie inconnue, d",
    "Nombre de jours de VM",
    "Procédure limitations de soins (LATA)",
]

death_imputed = PredictionTask(
    df=df_imputed,
    to_predict=to_predict,
    to_drop=to_drop_1
)

death_with_MV = PredictionTask(
    df=df_with_MV,
    to_predict=to_predict,
    to_drop=to_drop_1
)


# All tasks
tasks = {
    'death_imputed': death_imputed,
    'death_with_MV': death_with_MV,
}
