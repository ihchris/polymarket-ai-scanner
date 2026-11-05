"""Real-link lookup UI. Never substitutes demo data on lookup errors."""
import io
import json
import logging
import pandas as pd
import streamlit as st
from src.collectors.polymarket import PolymarketClient, Market
from src.collectors.market_link import lookup_link, inspect_market, refresh_selected_market, lookup_error_message
from src.collectors.historical import load_history
from src.analysis.linked_election import analyze_election_csv
from src.utils.validation import utc, utcnow


STATUS = {'RESOLVED': 'Resultado resolvido na Polymarket', 'UNRESOLVED': 'Disputa em aberto',
          'AWAITING_RESOLUTION': 'Encerrada, aguardando confirmação de resolução',
          'INCONSISTENT': 'Dados de resolução inconsistentes: não confirmar vencedor'}


def linked_market_panel(settings, persist):
    st.subheader('Analisar disputa pelo link — dados reais')
    st.caption('Consulta real mesmo com o scanner geral em DEMO. Um link identifica a disputa; não garante uma previsão de vencedor.')
    with st.form('market_link_form'):
        url = st.text_input('Link da disputa no Polymarket', placeholder='https://polymarket.com/event/...')
        submitted = st.form_submit_button('Analisar link')
    if submitted:
        # Clear previous results before a failed/new request to avoid showing an
        # old winner under the newly entered link.
        for key in ['linked_lookup', 'linked_report', 'linked_forecast', 'linked_selection', 'linked_market_id']:
            st.session_state.pop(key, None)
        client = PolymarketClient(retries=1)
        try:
            with st.spinner('Buscando a disputa na API oficial…'):
                context = lookup_link(client, url)
                st.session_state.linked_lookup = context
                if len(context['markets']) == 1:
                    st.session_state.linked_report = inspect_market(client, context['markets'][0])
        except Exception as exc:
            logging.getLogger(__name__).exception('Real market link lookup failed')
            st.error(lookup_error_message(exc))
        finally:
            client.close()
    context = st.session_state.get('linked_lookup')
    if not context:
        return
    st.caption(f'Link consultado: {context["url"]} • Consulta: {context["fetched_at"]}')
    if context['skipped']:
        st.warning(f'{context["skipped"]} disputas foram omitidas por dados inválidos na API.')
    markets = context['markets']
    by_id = {market.id: market for market in markets}
    chosen_id = st.selectbox('Disputa específica', list(by_id),
                            format_func=lambda mid: f'{by_id[mid].title} · ID {mid}', key='linked_market_id')
    chosen = by_id[chosen_id]
    if len(markets) > 1:
        st.caption(f'Este evento contém {len(markets)} disputas. Selecione o candidato/resultado que deseja verificar.')
    if st.button('Atualizar verificações e preços', key='verify_link'):
        st.session_state.pop('linked_report', None)
        st.session_state.pop('linked_forecast', None)
        client = PolymarketClient(retries=1)
        try:
            # Re-fetch metadata so resolution never relies on a stale cached
            # event snapshot while CLOB is queried again.
            market = refresh_selected_market(client, chosen.id, context['url'])
            with st.spinner('Conferindo resolução e livros de ofertas…'):
                st.session_state.linked_report = inspect_market(client, market)
        except Exception as exc:
            logging.getLogger(__name__).exception('Market verification failed')
            st.error(lookup_error_message(exc))
        finally:
            client.close()
    report = st.session_state.get('linked_report')
    if report is None or report['market']['id'] != chosen.id:
        st.info('Selecione a disputa e toque em Atualizar verificações e preços.')
        return
    market = Market.model_validate(report['market'])
    st.markdown(f'**{market.title}**')
    st.caption(f'Dados reais · ID {market.id} · Verificado em {report["checked_at"]}')
    if (utcnow()-utc(report['checked_at'])).total_seconds() > settings.maximum_age_seconds:
        st.warning('Esta consulta está antiga. Toque em Atualizar verificações e preços para obter o estado atual.')
    if report['status'] == 'RESOLVED':
        st.success(f'Resultado resolvido na Polymarket: {report["winner"]}')
        st.caption('Confirmação pelos campos explícitos de resolução Gamma/CLOB. Não constitui verificação independente do fato no mundo real.')
    else:
        st.warning(STATUS[report['status']])
        st.info('Quem vencerá: sem previsão independente automática disponível para esta disputa. O favorito do mercado não é resultado confirmado.')
    st.write(market.description or 'A API não forneceu descrição/regras de resolução.')
    st.write('Fonte de resolução informada pela Polymarket:', market.resolution_rules or 'Não informada')
    if report['checks']:
        st.dataframe(pd.DataFrame(report['checks']), use_container_width=True, hide_index=True)
    for warning in report['warnings']:
        st.warning(warning)
    if report['quotes']:
        quotes = pd.DataFrame(report['quotes'])
        quotes['age_seconds'] = (pd.Timestamp(utcnow()) - pd.to_datetime(quotes.timestamp, utc=True)).dt.total_seconds()
        quotes['dados_antigos'] = (quotes.age_seconds > settings.maximum_age_seconds) | (quotes.age_seconds < -5)
        st.dataframe(quotes, use_container_width=True, hide_index=True)
        if quotes.dados_antigos.any():
            st.warning('Há cotações antigas ou com timestamp futuro. Não use esses preços para avaliar vantagem atual.')
    if market.indicative_prices:
        st.dataframe(pd.DataFrame({'Outcome': market.outcomes, 'Preço indicativo (não é previsão independente)': market.indicative_prices}), hide_index=True)
    for name, source in report['sources'].items():
        if source:
            st.link_button(f'Fonte oficial: {name.upper()}', source)
    persist(f'linked-market:{market.id}:{report["checked_at"]}', 'markets', report, report['checked_at'])
    for side, book in market.books.items():
        persist(f'linked-book:{market.id}:{side}:{book.timestamp}', 'price_snapshots',
                {'market_id': market.id, 'side': side, 'source': 'LIVE_DATA', **book.model_dump(mode='json')}, book.timestamp)
    st.download_button('Baixar verificações JSON', json.dumps(report, indent=2), f'market-{market.id}.json', 'application/json')
    if not market.closed and {s.upper() for s in market.outcomes} == {'YES', 'NO'}:
        with st.expander('Previsão eleitoral com dados independentes fornecidos por você'):
            st.caption('Somente eleições de maioria simples com dois candidatos. Envie parciais municipais no esquema do README. O app valida os dados, mas não autentica automaticamente a fonte nem a correspondência com a disputa.')
            csv = st.file_uploader('CSV de resultados eleitorais reais', type='csv', key=f'linked_csv_{market.id}')
            if csv is not None:
                try:
                    content = csv.getvalue()
                    history = load_history(io.BytesIO(content))
                    times = sorted(history.timestamp.unique())
                    cutoff = st.selectbox('Usar somente informações disponíveis até', times, index=len(times)-1)
                    a = st.text_input('Nome do candidato A', key=f'candidate_a_{market.id}')
                    b = st.text_input('Nome do candidato B', key=f'candidate_b_{market.id}')
                    outcome = st.selectbox('Vitória do candidato A corresponde a', ['YES', 'NO'])
                    source_url = st.text_input('URL HTTPS da fonte dos dados eleitorais')
                    confirmed = st.checkbox('Revisei as regras: esta disputa corresponde à maioria simples entre estes dois candidatos, e o CSV contém dados reais desta eleição.')
                    if st.button('Calcular previsão com esses dados'):
                        st.session_state.pop('linked_forecast', None)
                        with st.spinner('Estimando a posterior com os dados enviados…'):
                            prediction = analyze_election_csv(market, content, cutoff, a, b, outcome,
                                                              source_url, confirmed, settings)
                        st.session_state.linked_forecast = prediction
                        persist(f'linked-forecast:{market.id}:{prediction["analyzed_at"]}', 'forecasts', prediction,
                                prediction['forecast']['timestamp'], prediction['forecast']['model_version'])
                except Exception as exc:
                    st.error(f'Previsão não calculada: {exc}')
            prediction = st.session_state.get('linked_forecast')
            if prediction and prediction['market_id'] == market.id:
                st.warning('PREVISÃO ESTATÍSTICA com dados do usuário. Não é resultado oficial nem garantia de vitória.')
                st.caption(f'Última previsão calculada em {prediction["analyzed_at"]}. Fonte informada: {prediction["source_url"]}')
                if (utcnow()-utc(prediction['forecast']['timestamp'])).total_seconds() > settings.maximum_age_seconds:
                    st.warning('Dados eleitorais antigos para o limite configurado; previsão histórica, não verificação atual.')
                st.write('Candidato com maior probabilidade no modelo:', prediction['probable_winner'])
                st.json(prediction['forecast'])
                st.dataframe(pd.DataFrame(prediction['opportunities']), use_container_width=True, hide_index=True)
                st.caption(prediction['data_validation'])
                st.download_button('Baixar previsão e rastreabilidade', json.dumps(prediction, indent=2), f'election-{market.id}.json', 'application/json')
