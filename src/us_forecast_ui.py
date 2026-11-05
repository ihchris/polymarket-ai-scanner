"""Separate pre-election US scenarios; never replaces the counting model."""
import hashlib
import io
import json
import logging
import pandas as pd
import plotly.express as px
import streamlit as st
from src.collectors.us_elections import MATCHUPS, fetch_public_scenario, demo_us_scenario, ALLOCATION_SOURCE
from src.models.us_presidential import forecast_us, PreElectionConfig
from src.utils.validation import utcnow
from src.collectors.market_link import lookup_error_message

POLL_COLUMNS = 'poll_id,unit,candidate_a,candidate_b,share_a,share_b,sample_size,pollster,end_date,available_at,election_year,source_url\n'
HISTORY_COLUMNS = 'unit,share_a,votes,year,available_at,source_url\n'


@st.cache_data(ttl=1800, show_spinner=False)
def public_inputs(race_id):
    return fetch_public_scenario(race_id)


def us_forecast_page(settings, persist):
    st.subheader('EUA — Pré-eleição')
    st.write('Pesquisas + histórico estadual + simulações conjuntas do Colégio Eleitoral. Use Election Forecast para o modelo durante a apuração.')
    st.info('Cada cenário assume que os dois candidatos escolhidos disputam a eleição geral. Não estima indicação partidária nem a chance incondicional de vencer entre todos os nomes da Polymarket.')
    source = st.radio('Fonte do modelo pré-eleitoral', ['Dados públicos reais', 'CSV do usuário', 'DEMO simulado'], key='us_source')
    bundle = None
    custom_a, custom_b = '', ''
    polls_file = history_file = None
    race_id = None
    if source == 'Dados públicos reais':
        race_id = st.selectbox('Confronto hipotético de 2028', list(MATCHUPS),
            format_func=lambda key: ' × '.join(MATCHUPS[key]), key='us_matchup')
        st.caption('Pesquisas nacionais: ElectionsUSA, com links para os levantamentos originais. Histórico 2024: compilação pública de Tony McGovern. Não são dados de apuração de 2028.')
        st.link_button('Dados e licença — ElectionsUSA', 'https://electionsusa.org/data/')
        st.link_button('Histórico — fonte e metodologia', 'https://github.com/tonmcg/US_County_Level_Election_Results_08-24')
        if st.button('Limpar cache das fontes públicas'):
            public_inputs.clear()
            st.success('Cache limpo. Calcule novamente para consultar as fontes.')
    elif source == 'CSV do usuário':
        st.caption('Pesquisas e histórico devem usar a mesma orientação de candidato/partido A. Proporções em 0–1, datas available_at com fuso UTC. Unidades: siglas dos estados/DC; distritos ME-1/ME-2 e NE-1/NE-2/NE-3; US para pesquisas nacionais.')
        polls_file = st.file_uploader('Pesquisas estaduais/nacionais CSV', type='csv', key='us_polls_csv')
        history_file = st.file_uploader('Histórico estadual/distrital CSV', type='csv', key='us_history_csv')
        custom_a = st.text_input('Candidato A (partido correspondente a share_a no histórico)', key='us_candidate_a')
        custom_b = st.text_input('Candidato B', key='us_candidate_b')
        st.download_button('Baixar cabeçalho CSV de pesquisas', POLL_COLUMNS, 'polls-template.csv', 'text/csv')
        st.download_button('Baixar cabeçalho CSV histórico', HISTORY_COLUMNS, 'history-template.csv', 'text/csv')
    else:
        st.warning('DEMO: histórico, candidatos e pesquisas inteiramente fictícios. Apenas a alocação de 538 eleitores é real.')
    with st.expander('Incerteza e parâmetros do cenário'):
        half_life = st.number_input('Meia-vida das pesquisas (dias)', 5., 365., 45., key='us_halflife')
        max_age = st.number_input('Idade máxima das pesquisas (dias)', 7, 1460, 365, key='us_maxage')
        common_sd = st.number_input('Erro eleitoral nacional compartilhado (SD logit)', .01, .5, .10, step=.01, key='us_common_sd')
        future_sd = st.number_input('Incerteza futura por raiz de ano (SD logit)', .01, 1., .20, step=.01, key='us_future_sd')
        st.caption('Escalas são hipóteses configuráveis, ainda sem calibração retrospectiva. A incerteza cresce com a distância até a eleição. Mudá-las muda a probabilidade calculada.')
    confirmed = True
    if source == 'CSV do usuário':
        confirmed = st.checkbox('Confirmei fonte, datas de disponibilidade, confronto e orientação partidária do histórico. Estes arquivos não contêm dados DEMO.', key='us_confirm')
    if st.button('Calcular cenário pré-eleitoral', type='primary', key='run_us_forecast'):
        st.session_state.pop('us_forecast_result',None)
        try:
            with st.spinner('Validando fontes e simulando o Colégio Eleitoral…'):
                if source == 'Dados públicos reais':
                    bundle = public_inputs(race_id)
                elif source == 'DEMO simulado':
                    bundle = demo_us_scenario()
                else:
                    if not confirmed or polls_file is None or history_file is None:
                        raise ValueError('Envie os dois arquivos e confirme a correspondência dos dados')
                    pbytes, hbytes = polls_file.getvalue(), history_file.getvalue()
                    bundle = {'polls': pd.read_csv(io.BytesIO(pbytes)), 'history': pd.read_csv(io.BytesIO(hbytes)),
                        'candidate_a': custom_a.strip(), 'candidate_b': custom_b.strip(), 'source': 'USER_CSV',
                        'warnings': [], 'attribution': 'Fonte declarada pelo usuário; autenticidade não verificada automaticamente',
                        'poll_sha256': hashlib.sha256(pbytes).hexdigest(), 'history_sha256': hashlib.sha256(hbytes).hexdigest()}
                as_of = bundle['fetched_at'] if bundle['source'] == 'DEMO' else utcnow().isoformat()
                config = PreElectionConfig(draws=settings.simulations, poll_half_life_days=half_life,
                    max_poll_age_days=max_age, common_election_error_sd=common_sd,
                    future_drift_sd_per_sqrt_year=future_sd)
                result = forecast_us(bundle['polls'], bundle['history'], bundle['candidate_a'],
                    bundle['candidate_b'], as_of=as_of, config=config, source=bundle['source'])
                result['provenance'] = {k:v for k,v in bundle.items() if k not in {'polls','history'}}
                result['history_used'] = bundle['history'].to_dict('records')
                st.session_state.us_forecast_result = result
                key = f'us-pre-election:{result["source"]}:{result["candidate_a"]}:{result["candidate_b"]}:{utcnow().isoformat()}'
                persist(key,'forecasts',result,result['as_of'],result['model_version'])
        except Exception as exc:
            logging.getLogger(__name__).exception('US pre-election forecast failed')
            message = lookup_error_message(exc) if not isinstance(exc,ValueError) else str(exc)
            st.error(f'Cenário não calculado: {message}. Nenhum dado simulado substituiu a fonte solicitada.')
    result = st.session_state.get('us_forecast_result')
    if not result:
        st.caption('Escolha uma fonte e toque em Calcular cenário pré-eleitoral. O modelo de apuração permanece no separador Election Forecast.')
        return
    # Keep every output labeled by its own source/matchup, even if inputs changed
    # without a new calculation; don't relabel an old DEMO as current public data.
    st.divider()
    st.markdown(f'**Último cenário calculado: {result["candidate_a"]} × {result["candidate_b"]}**')
    st.caption(f'Origem: {result["source"]} · Instante de corte: {result["as_of"]} · Versão: {result["model_version"]}')
    if result['source'] == 'DEMO':
        st.warning('RESULTADO DEMO — dados simulados, não previsão real de 2028.')
    elif result['source'] == 'USER_CSV':
        st.warning('CSV do usuário: estrutura validada; autenticidade das fontes não verificada automaticamente.')
    else:
        st.caption(result['provenance']['attribution'])
        st.caption('Dados de um agregador público, com links de fontes originais. Os documentos de cada pesquisa não foram auditados automaticamente pelo app.')
    a,b,c = st.columns(3)
    a.metric(f'{result["candidate_a"]}: P(≥270)',f'{result["probability_a_270"]:.1%}')
    b.metric(f'{result["candidate_b"]}: P(≥270)',f'{result["probability_b_270"]:.1%}')
    c.metric('Nenhum candidato chega a 270',f'{result["probability_no_majority"]:.1%}')
    lo,hi = result['electoral_votes_a_interval_95']
    st.write(f'Eleitores de {result["candidate_a"]}: média {result["electoral_votes_a_mean"]:.1f}; intervalo de 95% {lo:.0f}–{hi:.0f}.')
    st.caption('270 de 538 é a maioria. Empate/ausência de maioria é reportado separadamente; o modelo não escolhe vencedor via Câmara. Participação popular é entre os dois candidatos, com pesos históricos fixos, não previsão de comparecimento.')
    st.write(f'Participação popular projetada de A entre os dois candidatos: {result["popular_share_a_mean"]:.1%}.')
    for warning in result['warnings'] + result['provenance'].get('warnings',[]):
        st.warning(warning)
    hist = pd.DataFrame({'Eleitores A':range(539),'Cenários':result['ev_histogram_a']})
    fig = px.bar(hist[hist['Cenários']>0],x='Eleitores A',y='Cenários',title='Distribuição conjunta de votos no Colégio Eleitoral')
    fig.add_vline(x=270,line_dash='dash')
    st.plotly_chart(fig,use_container_width=True)
    units = pd.DataFrame(result['units'])
    states = units[~units.unit.str.contains('-')]
    st.plotly_chart(px.choropleth(states,locations='state',locationmode='USA-states',color='win_probability_a',
        scope='usa',range_color=[0,1],color_continuous_scale='RdBu',hover_name='name',
        title=f'Probabilidade condicional por estado — {result["candidate_a"]}'),use_container_width=True)
    st.dataframe(units,use_container_width=True,hide_index=True)
    st.caption(f'{result["poll_count"]} pesquisas usadas; {result["excluded_poll_count"]} excluídas por data, ciclo, confronto ou idade. Pesquisa mais recente: {result["latest_poll_end"]}. Estados sem pesquisa usam extrapolação histórica com incerteza.')
    with st.expander('Pesquisas utilizadas e links de origem'):
        st.dataframe(pd.DataFrame(result['polls_used']),use_container_width=True,hide_index=True)
    with st.expander('Hipóteses e diagnóstico'):
        st.json(result['diagnostics'])
        st.json(result['settings'])
        st.write('Modelo normal conjugado em escala logit com efeitos nacional, regional, estadual e de instituto. Sorteios conjuntos incluem erro nacional e deriva futura compartilhados. ME/NE têm dois eleitores estaduais e um por distrito; sem histórico distrital, o prior estadual recebe variância adicional. Probabilidades ainda não calibradas. Sem estimativa de primárias, terceiros, desistência ou voto infiel; não comparar diretamente com contratos incondicionais de todos os candidatos.')
    st.link_button('Regras e alocação oficial — National Archives',ALLOCATION_SOURCE)
    st.download_button('Baixar cenário, fontes e diagnóstico JSON',json.dumps(result,indent=2,default=str),
                       'us-pre-election-scenario.json','application/json')
