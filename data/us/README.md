# Modelo pré-eleitoral dos EUA

Esta pasta não contém pesquisas reais inventadas. O painel EUA — Pré-eleição
consulta fontes públicas no botão de cálculo, aceita CSVs do usuário ou gera
um DEMO explicitamente fictício com seed 2028.

Pesquisas CSV, uma linha por instituto/confronto/unidade:

poll_id,unit,candidate_a,candidate_b,share_a,share_b,sample_size,pollster,end_date,available_at,election_year,source_url

Histórico CSV, uma linha por unidade:

unit,share_a,votes,year,available_at,source_url

Proporções 0–1; votes é total dos dois partidos; histórico share_a deve
corresponder ao partido de candidate_a. Forneça os 50 estados e DC, mesmo ano
histórico, estritamente anterior ao ciclo projetado. Distritos ME-1/ME-2 e
NE-1/NE-2/NE-3 são opcionais; sua ausência aparece como aproximação com
incerteza extra. end_date pode ser data; available_at exige fuso UTC.
unit=US indica pesquisa nacional. CSVs reais exigem source_url HTTPS.
Não use data de encerramento da coleta como data de publicação quando
houver atraso: available_at deve refletir quando os dados ficaram disponíveis.

O cenário padrão é 2028. Este modelo é condicional a um confronto de dois
candidatos e não fornece a chance de cada candidato conquistar a indicação.
