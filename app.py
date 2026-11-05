"""Streamlit local dashboard. Analysis and simulation only."""
import io
import json
from datetime import timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from config.settings import Settings
from src.collectors.elections import DemoElection, BASE_TIME
from src.collectors.historical import load_history, as_of
from src.collectors.polymarket import PolymarketClient, demo_markets
from src.models.bayesian import forecast
from src.analysis.probability import from_forecast
from src.analysis.opportunities import evaluate
from src.analysis.backtest import replay
from src.storage.database import Database
from src.utils.validation import utcnow

st.set_page_config(page_title='Polymarket AI Scanner v2', page_icon='📊', layout='wide')
st.title('Polymarket AI Scanner v2')
st.caption('Análise e simulação • Sem execução financeira • LIVE_DATA consulta somente APIs públicas')
if 'settings' not in st.session_state:
    st.session_state.settings = Settings()
if 'progress' not in st.session_state:
    st.session_state.progress = 0.
    st.session_state.evolution = []
settings = st.session_state.settings
page = st.sidebar.radio('Página', ['Market Scanner', 'Election Forecast', 'Backtesting', 'Settings'])
st.sidebar.info(f'Modo: {settings.mode}\n\nMotor rápido: posterior de Laplace; NUTS disponível via CLI.')


@st.cache_data(show_spinner=False)
def get_forecast(progress, draws):
    return forecast(DemoElection().snapshot(progress), draws=draws)


@st.cache_data(show_spinner=False)
def get_snapshot(progress):
    return DemoElection().snapshot(progress)


def persist_once(key, table, payload, timestamp=None, version=None):
    if st.session_state.get(key):
        return
    db = Database(settings.database)
    try:
        db.append(table, payload, timestamp=timestamp, model_version=version)
    finally:
        db.close()
    st.session_state[key] = True


def record_forecast(f, snapshot, key):
    persist_once(f'partial:{key}', 'election_partials', snapshot.to_dict('records'), f.timestamp)
    persist_once(f'forecast:{key}', 'forecasts', f.to_dict(), f.timestamp, f.model_version)


if page == 'Settings':
    st.subheader('Configurações locais')
    with st.form('settings_form'):
        mode = st.selectbox('Modo', ['DEMO', 'LIVE_DATA'], index=['DEMO', 'LIVE_DATA'].index(settings.mode))
        refresh = st.number_input('Intervalo de atualização (s)', 5, 3600, settings.refresh_seconds)
        liquidity = st.number_input('Liquidez mínima (USD, filtro de metadados)', 0., value=settings.minimum_liquidity)
        edge = st.number_input('Vantagem mínima por contrato', 0., 1., settings.minimum_edge, step=.005)
        draws = st.number_input('Simulações', 500, 100000, settings.simulations, step=500)
        known_fees = st.checkbox('Configurei uma taxa efetiva após revisar o mercado', value=settings.estimated_fee_per_share is not None)
        fee = st.number_input('Custo de taxa por contrato (USD)', 0., 1., settings.estimated_fee_per_share or 0., step=.001, format='%.4f')
        slippage = st.number_input('Margem adicional de slippage por contrato (USD)', 0., 1., settings.slippage_per_share, step=.001, format='%.4f')
        age = st.number_input('Idade máxima dos dados (s)', 1, 86400, settings.maximum_age_seconds)
        quantity = st.number_input('Quantidade hipotética de contratos', .01, 100000., settings.order_size)
        if st.form_submit_button('Salvar configurações'):
            st.session_state.settings = Settings(mode=mode, refresh_seconds=refresh, minimum_liquidity=liquidity,
                minimum_edge=edge, simulations=draws, estimated_fee_per_share=fee if known_fees else None,
                slippage_per_share=slippage, maximum_age_seconds=age, order_size=quantity, database=settings.database)
            st.success('Configurações atualizadas nesta sessão.')
    st.info('DEMO usa taxa fictícia explícita de zero. Em LIVE_DATA, metadados de taxas são armazenados, mas não são convertidos em custo sem uma fórmula específica revisada. Nenhuma carteira é usada.')

elif page == 'Election Forecast':
    st.subheader('Election Forecast — eleição fictícia reproduzível')
    st.warning('DADOS SIMULADOS: 100 municípios, 3 estados, dois candidatos. A divulgação é não aleatória.')
    c1, c2, c3 = st.columns(3)
    if c1.button('Iniciar / reiniciar simulação'):
        st.session_state.progress = 0.
        st.session_state.evolution = []
    if c2.button('Avançar apuração (+10%)'):
        st.session_state.progress = round(min(1., st.session_state.progress + .1), 2)
    if c3.button('Concluir apuração'):
        st.session_state.progress = 1.
    progress = st.session_state.progress
    st.progress(progress, text=f'Tempo simulado: {progress:.0%} da janela de divulgação')
    with st.spinner('Atualizando posterior e simulando votos restantes…'):
        f = get_forecast(progress, settings.simulations)
    df = get_snapshot(progress)
    record_forecast(f, df, f'demo:{progress}:{settings.simulations}')
    if not any(r['timestamp'] == f.timestamp for r in st.session_state.evolution):
        st.session_state.evolution.append(f.to_dict())
    cols = st.columns(4)
    cols[0].metric('Votos apurados / participação estimada', f'{f.counted_votes_percentage:.1f}%')
    cols[1].metric('A: parcial oficial simulado', f'{df.candidate_a_votes.sum()/df.counted_votes.sum():.2%}' if df.counted_votes.sum() else 'Sem votos')
    cols[2].metric('A: percentual final projetado', f'{f.estimated_vote_share:.2%}')
    cols[3].metric('Probabilidade de vitória de A', f'{f.win_probability:.2%}')
    st.caption(f'IC posterior de 95% para participação de A: {f.credible_interval_95[0]:.2%} a {f.credible_interval_95[1]:.2%}. Versão {f.model_version}. Timestamp: {f.timestamp}')
    st.caption('Percentual apurado usa participação prevista como denominador; não é percentual de urnas nem amostra nacional aleatória. Em empate, a probabilidade divide a massa de empate igualmente, sem presumir regra jurídica.')
    hist = pd.DataFrame(st.session_state.evolution).sort_values('timestamp')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.timestamp, y=hist.credible_interval_95.map(lambda x: x[1]), line={'width': 0}, name='Limite superior'))
    fig.add_trace(go.Scatter(x=hist.timestamp, y=hist.credible_interval_95.map(lambda x: x[0]), fill='tonexty', line={'width': 0}, name='IC 95%'))
    fig.add_trace(go.Scatter(x=hist.timestamp, y=hist.estimated_vote_share, mode='lines+markers', name='Participação projetada A'))
    fig.add_hline(y=.5, line_dash='dash')
    st.plotly_chart(fig, use_container_width=True)
    st.plotly_chart(px.histogram(pd.DataFrame({'Participação final A': f.samples}), x='Participação final A', nbins=60), use_container_width=True)
    st.plotly_chart(px.line(hist, x='timestamp', y='win_probability', markers=True, title='Evolução da probabilidade de vitória de A'), use_container_width=True)
    st.dataframe(df, use_container_width=True, hide_index=True)
    with st.expander('Modelo, diagnóstico e limitações'):
        st.json(f.diagnostics)
        st.write('Logit binomial com swing nacional, urbano, por estado e município. Priors próprios e posterior de Laplace. A ordem de divulgação é ajustada sob hipótese condicional MAR. Um choque compartilhado nos votos restantes representa sensibilidade a seleção residual não identificável. Não corrige toda forma de viés de apuração; resultados reais exigem validação e covariáveis adequadas.')
    st.download_button('Baixar previsão JSON', json.dumps(f.to_dict(), indent=2), 'forecast-demo.json', 'application/json')

elif page == 'Market Scanner':
    st.subheader('Market Scanner')
    st.caption('Preços de compra usam asks e profundidade. Último negócio e preços indicativos não são usados para EV.')
    st.button('Atualizar agora')

    @st.fragment(run_every=f'{settings.refresh_seconds}s' if settings.mode == 'LIVE_DATA' else None)
    def scanner():
        markets = []
        fallback = settings.mode == 'DEMO'
        if not fallback:
            client = PolymarketClient()
            try:
                with st.spinner('Consultando Gamma e CLOB…'):
                    markets = client.markets(settings.minimum_liquidity, max_markets=10)
                if not markets or not any(m.books for m in markets):
                    raise ValueError('Nenhum livro de ofertas válido disponível')
                if client.warnings:
                    st.info(f'{len(client.warnings)} registros/livros não puderam ser usados. Demais dados continuam disponíveis.')
            except Exception as exc:
                fallback = True
                st.warning(f'LIVE_DATA indisponível ({type(exc).__name__}); exibindo DEMO claramente identificado.')
            finally:
                client.close()
        progress = st.session_state.progress
        f = None
        if fallback:
            now = BASE_TIME + timedelta(minutes=round(progress * 180))
            markets = demo_markets(progress, now)
            f = get_forecast(progress, settings.simulations)
            st.warning(f'DADOS SIMULADOS — relógio virtual: {now.isoformat()}')
        else:
            now = utcnow()
            st.info('Dados públicos reais. Nenhum mercado real é vinculado automaticamente ao modelo eleitoral fictício.')
        categories = sorted({m.category for m in markets})
        category = st.selectbox('Categoria', ['Todas'] + categories)
        filter_edge = st.checkbox('Mostrar somente vantagem líquida ≥ mínimo configurado', value=False)
        rows = []
        for m in markets:
            persist_once(f'market:{m.source}:{m.id}:{m.updated_at}', 'markets', m.model_dump(mode='json'), now)
            for side in ['YES', 'NO']:
                # Explicit identity mapping only for the synthetic election market.
                estimate = from_forecast(f, side) if f is not None and m.id == 'demo-0' else None
                row = evaluate(m, side, estimate, settings, now=now, fee_per_share=0 if m.source == 'DEMO' else None)
                book = m.books.get(side)
                row.update(category=m.category, bid=book.bid if book else None,
                    ask_price=book.ask if book else None, volume=m.volume, liquidity=m.liquidity,
                    book_timestamp=book.timestamp.isoformat() if book else None,
                    model='Eleição sintética' if estimate else 'Sem modelo disponível',
                    indicative_price=m.indicative_prices[m.outcomes.index(side)] if side in m.outcomes and m.indicative_prices else None)
                if book:
                    persist_once(f'book:{m.source}:{m.id}:{side}:{book.timestamp}', 'price_snapshots',
                                 {'market_id': m.id, 'side': side, 'source': m.source, **book.model_dump(mode='json')}, book.timestamp)
                persist_once(f'opportunity:{m.source}:{m.id}:{side}:{now}:{settings.model_dump_json()}', 'opportunities', row, now,
                             estimate.model_version if estimate else None)
                if category != 'Todas' and m.category != category:
                    continue
                if filter_edge and (row['net_edge'] is None or row['net_edge'] < settings.minimum_edge):
                    continue
                rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption('uncertainty = amplitude da faixa de sensibilidade de P(vitória) sob ±1 ponto percentual de viés sistemático. Não é um intervalo calibrado. POSSIBLE_EDGE é uma classificação analítica, sem recomendação automática.')
        with st.expander('Regras, tokens e metadados das taxas'):
            for market in markets:
                st.json(market.model_dump(mode='json'))
    scanner()

elif page == 'Backtesting':
    st.subheader('Backtesting cronológico')
    source = st.radio('Fonte de resultados', ['DEMO sintético', 'CSV histórico do usuário'])
    uploaded = None
    final_share = None
    if source == 'CSV histórico do usuário':
        uploaded = st.file_uploader('Resultados parciais CSV (esquema no README)', type='csv')
        final_share = st.number_input('Participação final observada de A (somente para avaliação)', 0., 1., .52, step=.001)
    price_file = st.file_uploader('Opcional: preços históricos executáveis CSV (timestamp, ask, depth, source)', type='csv')
    st.caption('Sem preços executáveis e taxa efetiva conhecida, retorno e drawdown ficam indisponíveis. Não se presume lucratividade. Replay simula no máximo uma entrada YES e liquidação final; drawdown considera somente saldo realizado.')
    if st.button('Executar backtesting'):
        try:
            with st.spinner('Reproduzindo atualizações sem acesso a dados futuros…'):
                if source == 'DEMO sintético':
                    demo = DemoElection()
                    history = pd.concat([demo.snapshot(p) for p in [0, .2, .4, .6, .8, 1]], ignore_index=True)
                    final_share = demo.outcome()['vote_share']
                else:
                    if uploaded is None:
                        raise ValueError('Selecione o CSV de resultados parciais')
                    history = load_history(uploaded)
                prices = pd.read_csv(price_file) if price_file is not None else None
                result = replay(history, sorted(history.timestamp.unique()), final_share,
                    draws=min(settings.simulations, 2000), price_history=prices,
                    fee_per_share=settings.estimated_fee_per_share, slippage=settings.slippage_per_share,
                    minimum_edge=settings.minimum_edge, quantity=settings.order_size,
                    maximum_age_seconds=settings.maximum_age_seconds)
            result['source'] = source
            st.session_state.backtest = result
            db = Database(settings.database)
            db.append('backtests', {'source': source, **result}, model_version='hierarchical-logit-v1')
            db.close()
        except Exception as exc:
            st.error(f'Backtesting não executado: {exc}')
    if 'backtest' in st.session_state:
        result = st.session_state.backtest
        st.caption(f'Fonte do replay exibido: {result["source"]}')
        st.json(result['metrics'])
        df = pd.DataFrame(result['forecasts'])
        st.plotly_chart(px.line(df, x='timestamp', y='probability', markers=True, title='Previsão de vitória de A'), use_container_width=True)
        shares = px.line(df, x='timestamp', y='vote_share', markers=True, title='Participação projetada versus resultado final')
        shares.add_hline(y=result['final_share'], line_dash='dash')
        st.plotly_chart(shares, use_container_width=True)
        calibration = pd.DataFrame(result['metrics']['calibration'])
        st.plotly_chart(px.scatter(calibration, x='predicted', y='observed', size='n', range_x=[0,1], range_y=[0,1], title='Calibração descritiva (instantes correlacionados)'), use_container_width=True)
        st.dataframe(df, use_container_width=True)
        if result['trades']:
            st.dataframe(pd.DataFrame(result['trades']), use_container_width=True)
        st.download_button('Baixar resultado JSON', json.dumps(result, indent=2), 'backtest.json', 'application/json')
