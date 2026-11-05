# Polymarket AI Scanner v2

MVP Python de **análise e simulação** de mercados de previsão, com dashboard
Streamlit, eleição sintética reproduzível, modelo bayesiano hierárquico,
Monte Carlo conjunto, SQLite e replay temporal. Nenhuma aposta é executada;
não há carteiras, chaves privadas, assinatura de ordens ou endpoints de negociação.

## Instalação e execução

Python 3.11+ (validado com Python 3.12). Em um terminal, dentro do repositório:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

No Windows, ative com `.venv\Scripts\activate`. Abra http://localhost:8501.
O dashboard inicia em DEMO e funciona **sem internet após a instalação**.
Não são necessárias credenciais. A primeira instalação requer internet ou wheels
locais. `.env.example` documenta variáveis opcionais; arquivos `.env` não são
carregados automaticamente: exporte as variáveis no shell, se desejado.

```bash
# Previsão offline em JSON, 10.000 cenários posteriores
python main.py --progress 0.5 --draws 10000
# Regenerar CSV simulado versionado (seed 42)
python main.py --export-demo --draws 1000
# Backend alternativo PyMC / NUTS (mais lento; requer compilador C/C++)
python main.py --progress 0.5 --draws 1000 --backend pymc
# Testes, incluindo os dois modelos e fallback offline
python -m pytest -q
```

## Uso das páginas

- **Market Scanner:** livros YES/NO, bid/ask, preço médio executável por quantidade,
  preços indicativos separados, volume, liquidez, EV, faixa de sensibilidade e
  timestamps. Filtros por categoria, liquidez nas configurações e vantagem.
  Atualização periódica em LIVE_DATA e botão de atualização manual.
- **Election Forecast:** inicie ou avance a eleição em passos de 10%; acompanhe
  parcial, projeção, IC de 95%, probabilidade de vitória e evolução temporal Plotly.
  Todos os resultados dessa página são explicitamente **SIMULADOS**.
- **EUA — Pré-eleição:** modelo separado de pesquisas, histórico por estado e
  Colégio Eleitoral. Fontes públicas, CSV do usuário ou DEMO, sem alterar a
  simulação de apuração. Consulte a seção específica abaixo.
- **Backtesting:** replay DEMO ou CSV histórico, rótulo final separado e CSV de
  preços executáveis opcional. Apresenta Brier, log loss, erro absoluto de votos,
  calibração descritiva e, se houver preços e taxas, uma operação hipotética.
- **Settings:** DEMO/LIVE_DATA, intervalo, liquidez, edge, quantidade, simulações,
  margem de slippage, taxa efetiva revisada e idade máxima. Configuração da sessão.

LIVE_DATA significa **somente leitura**. No scanner geral, falhas de API ou ausência de livros
válidos ativam DEMO com aviso. Na consulta por link não há substituição por DEMO.
Livros individuais inválidos são omitidos sem
substituir silenciosamente cotações reais por fictícias. Não há probabilidade
inventada para mercados sem modelo: a interface mostra **“Sem modelo disponível”**.
Somente o mercado fictício `demo-0` é associado automaticamente ao modelo eleitoral.
Mercados reais podem receber dados eleitorais do usuário pelo fluxo abaixo, com
mapeamento explícito e confirmação das regras; nunca recebem a eleição DEMO.
O núcleo `evaluate` também aceita `ProbabilityEstimate` independente, sem LLM.

## Analisar uma disputa real pelo link

1. Em **Market Scanner**, cole a URL em **Link da disputa no Polymarket** e toque
   em **Analisar link**. Funciona independentemente do modo do scanner geral.
2. Links `/event/slug`, `/event/slug/market-slug` e `/market/slug` são aceitos.
   Eventos com várias disputas exigem selecionar a disputa específica e tocar
   em **Atualizar verificações e preços**. Disputas únicas são verificadas de imediato.
   A seleção usa IDs estáveis, não objetos ou slugs. A atualização consulta
   `GET /markets/{id}`; se essa rota falhar, consulta novamente o evento original
   e localiza o mesmo ID. Não reaproveita metadados antigos para confirmar vencedor.
   Falhas exibem mensagens específicas para HTTP 403/404/429/5xx, timeout e rede.
3. O painel mostra regras, fonte de resolução informada, cotações, timestamps e
   verificações entre Gamma/CLOB. Atualize manualmente para reconsultar as APIs.
   **Não informa vencedor com base apenas em preço 1/0, favorito ou encerramento.**
4. Um resultado é marcado `RESOLVED` somente com encerramento concordante,
   tokens correspondentes, exatamente um campo `winner=true` no CLOB e ausência
   de status de resolução pendente informado no Gamma. Divergências geram
   `INCONSISTENT`; encerramento sem vencedor explícito fica aguardando resolução.
5. Baixe o JSON de verificações para guardar as fontes e o instante consultado.

Esse resultado é a resolução **na Polymarket**, não uma verificação externa
independente do fato. Gamma e CLOB são interfaces da mesma plataforma. Disputas
em aberto não têm um vencedor futuro verificável: o app informa quando não há
modelo independente. Não existe previsão universal de eleições, esportes,
cripto, notícias e outros eventos usando apenas uma URL.

Para eleições reais de maioria simples com dois candidatos, expanda **Previsão
eleitoral com dados independentes fornecidos por você**. Envie CSV no esquema
abaixo, escolha o instante de corte, informe nomes dos candidatos e URL da fonte,
associe a vitória de A a YES ou NO e confirme a correspondência com as regras.
O modelo processa somente dados disponíveis até esse instante. Nenhum dado DEMO
é injetado. A saída contém probabilidade, IC, candidato mais provável, custos/EV
quando suficientes, hash SHA-256 do CSV, fonte declarada e versão do modelo.

A autenticidade do CSV e da fonte declarada **não é verificada automaticamente**.
O painel deixa isso explícito. A previsão é uma estimativa sujeita às limitações
do modelo, não um resultado oficial nem garantia de vitória. Dados antigos e
timestamps futuros são tratados; regras de resolução exigem confirmação manual.
Esportes, eleições com outras regras e demais categorias precisam de modelos e
conectores próprios antes de oferecer previsões independentes.

## Modelo separado: EUA — Pré-eleição

Abra **EUA — Pré-eleição**, escolha **Dados públicos reais**, selecione um
confronto e toque em **Calcular cenário pré-eleitoral**. A página consulta dados
públicos gratuitos e exibe probabilidades condicionais de atingir 270 eleitores,
intervalo de votos no Colégio Eleitoral, mapa estadual, pesquisas utilizadas,
fontes, timestamps e diagnóstico. O modelo antigo continua em **Election Forecast**;
os estados de sessão dos dois modelos são independentes.

Fontes e crédito:

- Pesquisas: [ElectionsUSA](https://electionsusa.org/data/), APIs documentadas
  para Vance × Newsom, Vance × Ocasio-Cortez e Vance × Buttigieg. São dados de um
  agregador independente, não uma autoridade eleitoral. Compilação CC BY 4.0;
  campos derivados de Wikipedia CC BY-SA 4.0. Cada pesquisa preserva o link para
  sua fonte original; PDFs/documentos originais não são auditados pelo app.
- Histórico 2024: [compilação de Tony McGovern](https://github.com/tonmcg/US_County_Level_Election_Results_08-24),
  votos republicanos/democratas agregados por estado. É histórico de partido,
  não uma pesquisa de intenção de voto nos candidatos de 2028.
- Alocação: [National Archives](https://www.archives.gov/electoral-college/allocation),
  regras/quantidades vigentes para 2024 e 2028, 538 eleitores e maioria de 270.

Não precisa de chaves ou upload para esses três confrontos. Os conectores usam
HTTPS fixo, timeouts/retries e cache de 30 minutos. **Limpar cache das fontes
públicas** permite nova coleta. Falha da fonte gera erro, sem substituir dados
reais por DEMO. O pacote de saída inclui hashes SHA-256 dos arquivos consultados.
`available_at` das fontes públicas é conservadoramente o instante da coleta:
não se presume saber quando cada linha apareceu no agregador no passado.

### O que a probabilidade representa

O cenário estima **P(candidato A obter pelo menos 270 eleitores | A e B disputam
a eleição geral)**. Não inclui a probabilidade de indicação partidária,
desistência, terceiro candidato ou eleição contingente na Câmara. Não é a chance
incondicional de vencer entre todos os candidatos da Polymarket e não é ligada
automaticamente a contratos desses mercados ou usada para calcular EV neles.
Empate/ausência de maioria tem probabilidade separada; o modelo não decide o
vencedor nesse caso. O dashboard não garante um vencedor futuro.

### Estatística e incerteza

Modelo bayesiano normal conjugado na escala logit, com aproximação amostral das
pesquisas, distinto do binomial hierárquico da apuração:

```
logit(share_A_unit) = historical_unit_logit - historical_national_logit
                     + national_level + region_effect + state_effect
                     + district_effect_when_applicable
poll_logit ~ Normal(unit_or_national_level + pollster_effect, poll_variance)
```

As pesquisas usam A/(A+B), amostra efetiva ajustada pelo suporte aos dois nomes e
design effect padrão 1,5, piso de erro e desconto exponencial com meia-vida de
45 dias. Pesquisas repetidas são rejeitadas/deduplicadas na ingestão; efeitos de
instituto são compartilhados. Pesquisas de outro confronto/ciclo, publicadas ou
concluídas depois do cutoff ou mais antigas que o máximo configurado são excluídas.

Priors SD logit padrão: nível nacional 0,40; região 0,10; estado 0,15; instituto
0,08; distrito 0,12 quando houver histórico específico ou 0,35 quando ausente.
O nível nacional é centrado no histórico; demais efeitos são centrados em zero.
Cada cenário Monte Carlo sorteia conjuntamente todos os parâmetros. Acrescenta
erro eleitoral nacional compartilhado (SD 0,10) e deriva futura nacional
(SD 0,20 × raiz dos anos até a eleição), mais deriva regional e estadual.
Esses valores são **hipóteses configuráveis, ainda não calibradas por backtesting
de múltiplas eleições**, e não estimativas garantidas de erro real.

Estados sem pesquisa são extrapolados do perfil partidário histórico com incerteza.
O conector público atual fornece pesquisas **nacionais**, não inventa pesquisas
estaduais: o painel alerta que as projeções estaduais são extrapolações. Estados
recebem seus eleitores winner-take-all, exceto Maine e Nebraska: dois eleitores
estaduais mais um por distrito. Na ausência de histórico distrital, o prior vem
do estado com variância distrital extra, hipótese explicitada na tabela. Pesos
de votos populares são fixos no histórico, não um modelo dinâmico de participação.

### CSV e DEMO

**CSV do usuário** aceita outros confrontos e pesquisas estaduais/distritais.
Baixe os cabeçalhos no painel; esquema em [data/us/README.md](data/us/README.md).
Forneça 50 estados e DC no histórico, mesmo ano anterior ao ciclo projetado,
`available_at` com fuso e URLs das fontes. Os votos/share_A históricos devem
corresponder ao partido de A no confronto. Distritos são opcionais, mas, quando
fornecidos, precisam do mesmo ciclo histórico. Fontes declaradas pelo usuário
são validadas estruturalmente, não autenticadas automaticamente.

**DEMO simulado** funciona offline, com seed 2028, nomes/histórico/pesquisas
inteiramente fictícios e identificados. Apenas a alocação eleitoral é factual.
Resultados dos três modos permanecem rotulados pela fonte e confronto originais,
mesmo se o usuário mudar inputs sem recalcular. O JSON exporta parâmetros,
pesquisas usadas, histórico, proveniência e diagnóstico; o SQLite preserva a
versão separada `us-pre-election-gaussian-v1`.

## Modelo estatístico independente

A discussão https://www.reddit.com/r/brasil/s/c65bc7cBby é referência conceitual
fornecida no escopo, não fonte de código ou metodologia reproduzida literalmente.
Não se presume acesso ao código do autor. Esta implementação é independente.

Para município m no estado s:

```
logit(p_m) = logit(previous_share_m) + national_swing
             + urban_m * urban_swing + state_swing_s + local_swing_m
candidate_a_counted_m ~ Binomial(counted_valid_votes_m, p_m)
```

Priors normais próprios, centrados em zero, com desvios na escala logit:
nacional 0,30; coeficiente urbano 0,20; estado 0,20; município 0,15.
O pooling ocorre por parâmetros nacionais e estaduais compartilhados. Os desvios
são **fixos como hiperparâmetros de MVP**, não estimados como se houvesse várias
eleições de treinamento. `previous_share` deve ser independente da eleição atual.
A modelagem é de dois candidatos, votos válidos e maioria simples; não cobre
segundo turno, coligações, outros candidatos ou regras eleitorais diferentes.

O backend padrão encontra a moda posterior com gradiente/Hessiana analíticos e
amostra uma aproximação normal conjunta (Laplace). É bayesiano aproximado,
**não MCMC**. `--backend pymc` amostra o mesmo modelo com duas cadeias NUTS e
750 passos de aquecimento; informa R-hat, ESS e divergências via ArviZ.
Revise convergência antes de uso real. Não existe garantia de calibração.

Cada linha de Monte Carlo é **um cenário posterior conjunto**: nacional, estado
e município são compartilhados corretamente antes da agregação. Não há sorteio
independente de uma probabilidade nacional para cada município. A participação
restante usa choque lognormal nacional (SD 0,04) e local (SD 0,03), limitada pelo
eleitorado e votos já contados. Municípios concluídos têm votos finais fixos.
Votos restantes são binomiais condicionais às probabilidades daquele cenário.
Intervalos de credibilidade são quantis dos percentuais finais; P(vitória) é a
fração de cenários com maioria. Empates dividem a massa igualmente entre os dois
lados, convenção analítica que deve ser adaptada às regras reais.

### Divulgação não aleatória e limitações

No DEMO, a velocidade de divulgação depende sistematicamente do estado e do
perfil urbano. O modelo recebe esses atributos para **todos** os municípios,
inclusive sem votos. Não transforma percentual nacional de apuração em amostra
aleatória. Ajusta seleção sob hipótese de ignorabilidade condicional (MAR) dadas
as covariáveis. Dentro de cada município, votos já divulgados são assumidos
intercambiáveis condicionais ao modelo; isso pode falhar em dados reais.

Seleção residual MNAR não é identificável a partir das parciais. Um choque
normal compartilhado de SD 0,08 logit, ponderado pela fração restante, propaga
sensibilidade adicional; não é uma correção comprovada do viés. A taxa de
participação não é inferida de um arquivo de eleitores ausente: `expected_votes`
é uma previsão fornecida externamente, e `electorate` é o limite conhecido.
Latitude/longitude são preservadas como metadados; a correlação geográfica é
representada pelo estado, sem processo espacial contínuo neste MVP.
Resultados reais exigem fontes oficiais, covariáveis adequadas e validação
retrospectiva de priors, participação, seleção e regras de resolução.

O DEMO contém 100 municípios e três estados, seed fixa, eleição anterior,
turnout, resultado final privado e ordem de divulgação não aleatória. O modelo
só recebe o snapshot público. Não recebe `final_votes`, `final_a` nem o gerador.
A proporção de votos apurados usa participação prevista (ou votos finais dos
municípios concluídos), não contagem nacional de urnas como amostra aleatória.

## APIs oficiais e preços

Cliente HTTPX limitado a GET, com timeout de 12 s, até três retries adicionais,
backoff exponencial e respeito a Retry-After numérico (limitado a 30 s).
Paginação Gamma limitada a 500 metadados por atualização; dashboard consulta
até dez mercados binários elegíveis para limitar carga. Erros e dados inválidos
não geram probabilidades ou ordens artificiais.

- Gamma `GET https://gamma-api.polymarket.com/markets`, filtros active/closed,
  limit/offset: https://docs.polymarket.com/api-reference/markets/list-markets
- CLOB `GET https://clob.polymarket.com/book?token_id=...`:
  https://docs.polymarket.com/api-reference/orderbook/get-order-book-summary
- CLOB `GET https://clob.polymarket.com/fee-rate?token_id=...`:
  https://docs.polymarket.com/developers/CLOB/introduction
- Consulta por link: Gamma `GET /events/slug/{slug}`, `GET /markets/slug/{slug}`;
  resolução explícita via CLOB `GET /markets/{condition_id}`. Os links de fontes
  efetivamente consultadas aparecem no JSON e no painel. URLs do usuário não são
  requisitadas diretamente: somente slugs validados e APIs oficiais fixas são usados.

`outcomes` e `clobTokenIds` são mapeados por posição; JSON em strings é validado.
`outcomePrices` é indicativo e nunca substitui ask. CLOB timestamp em milissegundos
é preservado; timestamp de coleta é separado. Regras incluem descrição e fonte
de resolução. Os metadados de taxas retornados são armazenados sem presumir
que uma taxa-base seja um custo linear constante por contrato.

## EV e classificação

Para contrato que paga US$1 e quantidade q:

```
gross_edge = P(outcome) - best_ask
execution_price = VWAP das asks suficientes para q
estimated_costs = (execution_price - best_ask) + effective_fee + slippage_margin
net_edge = gross_edge - estimated_costs
```

Spread é mostrado, mas não somado novamente: o preço ask já incorpora o lado de
compra. A liquidez Gamma é filtro preliminar; a profundidade real é verificada
no livro, sem inventar fill para quantidade maior que a disponível. Taxa efetiva
por contrato é configuração explícita após revisão da fórmula vigente, enquanto
DEMO usa taxa fictícia de zero. Slippage é margem adicional; movimento de mercado,
latência ou livro desatualizado podem impedir uma execução hipotética.

- `INSUFFICIENT_DATA`: modelo independente, taxas, livros ou profundidade ausentes;
  mercado encerrado, inativo ou liquidez abaixo do mínimo.
- `STALE`: cotação, metadados ou previsão antigos; timestamps futuros são rejeitados.
- `NO_EDGE`: EV líquido não positivo.
- `UNCERTAIN`: resolução ambígua ou vantagem não robusta à incerteza.
- `POSSIBLE_EDGE`: EV conservador supera o mínimo configurado, em simulação.

A faixa conservadora de P(vitória) usa sensibilidade de ±1 ponto percentual no
percentual final de votos em todos os cenários. É uma faixa de cenários,
**não IC bayesiano de P(vitória)** nem erro Monte Carlo. Não demonstra edge real.
Mercados reais ficam conservadoramente marcados com resolução pendente de revisão.
Nenhuma classificação constitui recomendação automática de aposta.

## CSV e replay sem vazamento temporal

`data/demo/partials.csv` é **inteiramente fictício** e reproduzível.
`data/demo/outcome.json` fica separado. Esquema de parciais, uma linha por
município/instante:

```
municipality,state,previous_share,expected_votes,electorate,urban,latitude,longitude,counted_votes,candidate_a_votes,complete,timestamp
```

`latitude` e `longitude` são opcionais; demais colunas são obrigatórias.
`previous_share` em (0,1); `urban` é covariável numérica com escala validada pelo
usuário; contagens são inteiras não negativas, candidato <= total <= eleitorado.
`complete` deve ser True/False (ou 1/0); timestamp ISO 8601 com fuso, preferencialmente
UTC. Inclua uma linha inicial com zero votos para **cada** município. Não preencha
`expected_votes` usando participação final conhecida somente depois.

Em cada cutoff, o replay filtra timestamp <= cutoff e seleciona a última linha
por município. O rótulo final só entra no cálculo de métricas após todas as
previsões. Preços também são selecionados as-of, respeitando idade e profundidade;
preços futuros não podem preencher operações anteriores.
CSV opcional de preços: `timestamp,ask,depth,source`. Forneça preço executável
para a quantidade configurada (ou VWAP calculado para ela), nunca último negócio.
Não há coletor automático de preços históricos executáveis neste MVP.

Sem preços e taxa conhecida, retorno, lucro e drawdown ficam nulos. Havendo dados,
o simulador abre **no máximo uma posição YES por eleição**, somente antes de
apuração completa, usa quantidade fixa e liquida no resultado final. Retorno é
lucro/custo, drawdown em USD é sobre saldo **realizado**, capital inicial US$1000;
não representa drawdown de marcação a mercado nem carteira histórica real.
A simulação não reivindica lucratividade. Observações sucessivas da mesma eleição
são correlacionadas: calibração, Brier e log loss ao longo de uma eleição são
apenas descritivos; validação requer várias eleições independentes.

## Persistência e arquitetura

SQLite em `data/scanner.sqlite3`, escrita append-only, journal `DELETE` e timestamps UTC.
O journal tradicional evita exigir suporte a WAL/shared memory do sistema de
arquivos de hospedagem. Conexões aguardam bloqueios por até 30 segundos e são
fechadas se a inicialização falhar.
Tabelas separadas: markets, price_snapshots, election_partials, forecasts,
opportunities, backtests. Previsões e avaliações preservam versão do modelo.
Novas observações não sobrescrevem snapshots históricos; a interface evita
inserir duplicatas idênticas na mesma sessão. Settings ficam na sessão Streamlit.
Defina `SCANNER_DATABASE` para outro arquivo.

No dashboard, se o arquivo configurado não puder ser aberto ou escrito, novos
snapshots passam para um SQLite temporário por sessão, com aviso visível. O
histórico anterior permanece no arquivo original; não é migrado automaticamente.
Se até o armazenamento temporário falhar, a análise continua com aviso de que
novos snapshots não estão sendo salvos. Esse tratamento também cobre previsões
e backtesting. Dados temporários podem desaparecer ao reiniciar o Streamlit Cloud;
baixe os resultados importantes. Detalhes técnicos da falha ficam nos logs.

```
app.py                     cinco páginas Streamlit / Plotly
main.py                    CLI DEMO, exportação e backend opcional PyMC
config/settings.py         configuração validada
src/collectors/            Gamma/CLOB, eleição sintética e leitura histórica
src/models/                GLMM de apuração, modelo pré-eleitoral EUA e Monte Carlo
src/analysis/              probabilidades, EV, risco e replay
src/storage/database.py    snapshots SQLite
src/utils/validation.py    validação e UTC
 data/demo/                fixtures claramente simuladas
 data/us/                  esquema de pesquisas/histórico pré-eleitoral
 data/historical/          instruções para arquivos do usuário
 tests/                    parsing, erros HTTP, EV, temporalidade, modelo e UI
```

FastAPI foi omitido por não ser necessário ao dashboard local. Não há serviços
pagos, LLM na modelagem, fontes históricas reais inventadas ou execução financeira.

## Verificação desta implementação

- 90 testes passaram, incluindo navegação nas cinco páginas, avanço do DEMO,
  fallback offline, parsing Gamma/CLOB, custos/profundidade, determinismo,
  pooling, conclusão da eleição, isolamento de rótulos/preços futuros,
  orçamento, persistência append-only, journal compatível, fechamento após erro,
  fallback de diretório sem escrita e funcionamento sem qualquer armazenamento.
- Consulta por link cobre validação de domínio/caminho, seleção de disputas,
  resolução explícita, preços 1/0 sem vencedor, tokens divergentes, resultado
  pendente, ausência de fallback fictício e uso de CSV sem vazamento temporal.
- Evento real `presidential-election-winner-2028` consultado: 128 disputas;
  atualização por ID e livros YES/NO de duas disputas funcionaram. Testes locais
  cobrem troca de candidato, fallback ao evento e mensagens seguras de erro.
- Modelo pré-eleitoral cobre alocação de 538 eleitores, distritos ME/NE,
  probabilidade de ausência de maioria, determinismo, efeito das pesquisas,
  filtros de publicação/ciclo/confronto, duplicatas, histórico futuro/misto,
  ingestão de fontes públicas e preservação dos dois modelos no dashboard.
- Os três confrontos públicos de 2028 foram consultados e calculados; os dados
  atuais disponíveis são nacionais. A execução real não comprova calibração.
- `streamlit run app.py` iniciou; `/_stcore/health` retornou HTTP 200.
- Uma consulta pública real retornou um mercado com livros YES/NO e metadados
  de taxas, sem qualquer escrita nas APIs.
- Smoke test PyMC de 500 amostras totais executou com zero divergências,
  R-hat máximo 1,03 e ESS mínimo 139. A execução curta não satisfaz o alvo
  usual de R-hat <= 1,01; aumente amostras e revise diagnóstico antes de
  interpretar resultados reais. Essa verificação não demonstra calibração.
