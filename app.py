import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

# Импорт наших модулей
try:
    from data_provider import DataProvider
    from strategies import TradingStrategies
    from indicators import TechnicalIndicators
    from dominance_strategy import DominanceCorrelationStrategy
    from backtester import Backtester
    from position_tracker import PositionTracker
    st.success("✅ Все модули загружены успешно!")
except ImportError as e:
    st.error(f"❌ Ошибка импорта: {e}")
    st.stop()

# Конфигурация страницы
st.set_page_config(
    page_title="🚀 Trading Bot Pro - 15 Coins",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS стили
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .signal-card {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 5px solid;
    }
    
    .signal-strong-buy {
        background: linear-gradient(135deg, #00c851 0%, #007e33 100%);
        border-left-color: #00ff00;
        color: white;
    }
    
    .signal-buy {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        border-left-color: #28a745;
        color: white;
    }
    
    .signal-strong-sell {
        background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
        border-left-color: #ff0000;
        color: white;
    }
    
    .signal-sell {
        background: linear-gradient(135deg, #fd7e14 0%, #e83e8c 100%);
        border-left-color: #fd7e14;
        color: white;
    }
    
    .signal-hold {
        background: linear-gradient(135deg, #6c757d 0%, #adb5bd 100%);
        border-left-color: #6c757d;
        color: white;
    }
    
    .risk-levels {
        background: rgba(255,255,255,0.1);
        padding: 0.5rem;
        border-radius: 5px;
        margin-top: 0.5rem;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

class EnhancedTradingDashboard:
    """Улучшенный торговый дашборд с трекером позиций и рекомендациями"""
    
    def __init__(self):
        self.data_provider = DataProvider()
        self.strategies = TradingStrategies()
        self.indicators = TechnicalIndicators()
        self.backtester = Backtester(initial_balance=10000)
        self.position_tracker = PositionTracker()
        
        # Расширенный список криптовалют (15 монет)
        self.available_symbols = [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT',
            'ADA/USDT', 'DOGE/USDT', 'SOL/USDT', 'AVAX/USDT',
            'DOT/USDT', 'LINK/USDT', 'UNI/USDT', 'LTC/USDT',
            'BCH/USDT', 'ATOM/USDT', 'ALGO/USDT'
        ]

        # Названия монет для отображения
        self.symbol_names = {
            'BTC/USDT': '₿ Bitcoin',
            'ETH/USDT': 'Ξ Ethereum', 
            'BNB/USDT': '🔸 Binance Coin',
            'XRP/USDT': 'Ʀ Ripple',
            'ADA/USDT': '₳ Cardano',
            'DOGE/USDT': '🐕 Dogecoin',
            'SOL/USDT': '◎ Solana',
            'AVAX/USDT': '🔺 Avalanche',
            'DOT/USDT': '● Polkadot',
            'LINK/USDT': '🔗 Chainlink',
            'UNI/USDT': '🦄 Uniswap',
            'LTC/USDT': '🔷 Litecoin',
            'BCH/USDT': '🟠 Bitcoin Cash',
            'ATOM/USDT': '⚪ Cosmos',
            'ALGO/USDT': '🔵 Algorand'
        }
        
        # Инициализация состояния
        if 'monitoring_active' not in st.session_state:
            st.session_state.monitoring_active = False
        if 'monitored_symbols' not in st.session_state:
            st.session_state.monitored_symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
        if 'update_interval' not in st.session_state:
            st.session_state.update_interval = 30  # 30 секунд по умолчанию
        if 'selected_strategies' not in st.session_state:
            st.session_state.selected_strategies = ['mean_reversion', 'trend_following', 'breakout']
        if 'last_update_time' not in st.session_state:
            st.session_state.last_update_time = 0
        if 'quick_entry_data' not in st.session_state:
            st.session_state.quick_entry_data = None
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = []
        if 'min_signal_confidence' not in st.session_state:
            st.session_state.min_signal_confidence = 68  # НОВЫЙ ФИЛЬТР
    
    def get_available_strategies(self) -> Dict:
        """Получение всех доступных стратегий"""
        return {
            'mean_reversion': {
                'name': '🎯 Mean Reversion',
                'description': 'Покупка на отскоках от экстремумов',
                'details': 'RSI + Bollinger Bands для точных входов'
            },
            'trend_following': {
                'name': '📈 Trend Following',
                'description': 'Следование за трендом',
                'details': 'EMA + MACD + ADX для подтверждения'
            },
            'breakout': {
                'name': '🚀 Volatility Breakout',
                'description': 'Торговля на пробоях волатильности',
                'details': 'Bollinger Squeeze + Volume Spike'
            },
            'scalping': {
                'name': '⚡ Scalping',
                'description': 'Быстрые сделки на коротких движениях',
                'details': 'Fast EMA + RSI для скальпинга'
            },
            'momentum': {
                'name': '🔥 Momentum',
                'description': 'Торговля на сильных импульсах',
                'details': 'Momentum индикатор + Volume'
            }
        }
    
    def classify_signal_strength(self, confidence: float, strategy_name: str) -> Tuple[str, str, str]:
        """Классификация силы сигнала"""
        
        if confidence >= 75:
            return "СИЛЬНЫЙ", "🟢", "risk-high"
        elif confidence >= 60:
            return "СРЕДНИЙ", "🟡", "risk-medium"
        else:
            return "СЛАБЫЙ", "⚪", "risk-low"
    
    def calculate_signal_levels(self, price: float, direction: str, strategy: str) -> Dict:
        """Расчет рекомендуемых уровней стоп-лосса и тейк-профита для сигнала"""
        
        # Базовые параметры риск-менеджмента в зависимости от стратегии
        risk_params = {
            'mean_reversion': {'stop_pct': 2.5, 'tp_pct': 5.0},     # 1:2
            'trend_following': {'stop_pct': 3.0, 'tp_pct': 9.0},   # 1:3
            'breakout': {'stop_pct': 2.0, 'tp_pct': 6.0},          # 1:3
            'scalping': {'stop_pct': 1.5, 'tp_pct': 3.0},          # 1:2
            'momentum': {'stop_pct': 3.5, 'tp_pct': 7.0},          # 1:2
            'manual': {'stop_pct': 3.0, 'tp_pct': 6.0}             # По умолчанию
        }
        
        params = risk_params.get(strategy, risk_params['manual'])
        
        if direction == 'LONG' or direction == 'BUY':
            stop_loss = price * (1 - params['stop_pct'] / 100)
            take_profit = price * (1 + params['tp_pct'] / 100)
        else:  # SHORT или SELL
            stop_loss = price * (1 + params['stop_pct'] / 100)
            take_profit = price * (1 - params['tp_pct'] / 100)
        
        risk_reward = params['tp_pct'] / params['stop_pct']
        
        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'stop_pct': params['stop_pct'],
            'tp_pct': params['tp_pct'],
            'risk_reward': risk_reward
        }
    
    def analyze_symbol_enhanced(self, symbol: str, timeframe: str = '15m') -> Dict:
        """Расширенный анализ символа всеми стратегиями"""
        
        # Загружаем данные
        data = self.data_provider.get_market_data(symbol, interval=timeframe, limit=200)
        
        if data is None or data.empty:
            return {'error': f'Не удалось загрузить данные для {symbol}'}
        
        # Текущая цена и статистика
        current_price = data['close'].iloc[-1]
        price_change_24h = ((current_price / data['close'].iloc[-24]) - 1) * 100 if len(data) >= 24 else 0
        
        # Анализ рынка
        market_conditions = self.strategies.analyze_market_conditions(data)
        
        # Получаем сигналы от выбранных стратегий
        all_signals = self.strategies.combine_signals(data, st.session_state.selected_strategies)
        
        # Фильтруем по времени (только свежие сигналы)
        recent_signals = []
        if all_signals:
            cutoff_time = data.index[-1] - timedelta(hours=6)
            for signal in all_signals:
                signal_time = signal['timestamp']
                if isinstance(signal_time, str):
                    signal_time = pd.to_datetime(signal_time)
                elif hasattr(signal_time, 'tz_localize'):
                    signal_time = signal_time.tz_localize(None) if signal_time.tz else signal_time
                
                if signal_time >= cutoff_time:
                    # Добавляем рекомендуемые уровни к каждому сигналу
                    direction = signal['type']
                    strategy = signal.get('strategy', 'manual')
                    levels = self.calculate_signal_levels(signal['price'], direction, strategy)
                    
                    signal['recommended_stop_loss'] = levels['stop_loss']
                    signal['recommended_take_profit'] = levels['take_profit']
                    signal['risk_reward_ratio'] = levels['risk_reward']
                    signal['stop_pct'] = levels['stop_pct']
                    signal['tp_pct'] = levels['tp_pct']
                    
                    recent_signals.append(signal)
        
        # Определение консенсуса
        buy_signals = len([s for s in recent_signals if s['type'] == 'BUY'])
        sell_signals = len([s for s in recent_signals if s['type'] == 'SELL'])
        
        if buy_signals > sell_signals * 1.5:
            consensus = 'BUY'
            consensus_strength = (buy_signals / (buy_signals + sell_signals)) * 100 if (buy_signals + sell_signals) > 0 else 0
        elif sell_signals > buy_signals * 1.5:
            consensus = 'SELL'
            consensus_strength = (sell_signals / (buy_signals + sell_signals)) * 100 if (buy_signals + sell_signals) > 0 else 0
        else:
            consensus = 'HOLD'
            consensus_strength = 50
        
        # Средняя уверенность
        avg_confidence = np.mean([s.get('confidence', 50) for s in recent_signals]) if recent_signals else 0
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'price_change_24h': price_change_24h,
            'data': data,
            'market_conditions': market_conditions,
            'signals': recent_signals,
            'consensus': consensus,
            'consensus_strength': consensus_strength,
            'avg_confidence': avg_confidence,
            'active_signals': len(recent_signals),
            'buy_signals': buy_signals,
            'sell_signals': sell_signals
        }
    
    def group_signals_by_symbol(self, analysis_results: List[Dict]) -> Dict:
        """Группировка сигналов по символам - один лучший сигнал на монету С ФИЛЬТРОМ 68%+"""
        
        grouped_signals = {}
        
        for result in analysis_results:
            if 'error' not in result and result['signals']:
                symbol = result['symbol']
                symbol_name = self.symbol_names.get(symbol, symbol)
                
                # ФИЛЬТРУЕМ СИГНАЛЫ ПО МИНИМАЛЬНОЙ УВЕРЕННОСТИ
                qualified_signals = [
                    signal for signal in result['signals'] 
                    if signal.get('confidence', 0) >= st.session_state.min_signal_confidence
                ]
                
                if not qualified_signals:
                    continue  # Пропускаем если нет сигналов с нужной уверенностью
                
                # Находим лучший сигнал среди отфильтрованных
                best_signal = None
                best_confidence = 0
                
                for signal in qualified_signals:
                    confidence = signal.get('confidence', 0)
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_signal = signal
                
                if best_signal:
                    strength, color, css_class = self.classify_signal_strength(
                        best_signal.get('confidence', 50), best_signal.get('strategy', 'unknown'))
                    
                    grouped_signals[symbol] = {
                        'symbol': symbol,
                        'symbol_name': symbol_name,
                        'strategy': best_signal.get('strategy', 'unknown'),
                        'action': best_signal['type'],
                        'strength': strength,
                        'color': color,
                        'css_class': css_class,
                        'confidence': best_signal.get('confidence', 50),
                        'price': best_signal['price'],
                        'current_price': result['current_price'],
                        'reason': best_signal.get('reason', 'Signal detected'),
                        'timestamp': best_signal['timestamp'],
                        'consensus': result['consensus'],
                        'total_signals': len(result['signals']),
                        # Добавляем рекомендации по уровням
                        'recommended_stop_loss': best_signal.get('recommended_stop_loss'),
                        'recommended_take_profit': best_signal.get('recommended_take_profit'),
                        'risk_reward_ratio': best_signal.get('risk_reward_ratio'),
                        'stop_pct': best_signal.get('stop_pct'),
                        'tp_pct': best_signal.get('tp_pct')
                    }
        
        return grouped_signals
    
    def render_enhanced_controls(self):
        """Улучшенная панель управления"""
        
        st.sidebar.markdown("""
        <div class="main-header">
            <h2>🚀 Trading Bot Pro</h2>
            <p>15 монет • 5 стратегий • Трекер позиций</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Управление мониторингом
        st.sidebar.header("🔄 Мониторинг")
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("▶️ Старт" if not st.session_state.monitoring_active else "⏸️ Стоп", key="monitoring_toggle"):
                st.session_state.monitoring_active = not st.session_state.monitoring_active
                if not st.session_state.monitoring_active:
                    st.sidebar.info("⏸️ Мониторинг остановлен")
        
        with col2:
            if st.button("🔄 Обновить", key="manual_refresh"):
                st.session_state.last_update_time = 0  # Принудительное обновление
                st.rerun()
        
        # Настройки
        st.session_state.update_interval = st.sidebar.selectbox(
            "Интервал обновления:",
            [10, 30, 60, 120, 300],
            index=1,  # 30 секунд по умолчанию
            format_func=lambda x: f"{x} сек" if x < 60 else f"{x//60} мин"
        )
        
        # НОВЫЙ ФИЛЬТР ПО УВЕРЕННОСТИ СИГНАЛОВ
        st.sidebar.header("🎯 Фильтр сигналов")
        st.session_state.min_signal_confidence = st.sidebar.slider(
            "Минимальная уверенность сигнала:",
            min_value=50,
            max_value=95,
            value=68,
            step=5,
            help="Показывать только сигналы с уверенностью выше указанного процента"
        )
        
        # Выбор монет
        st.sidebar.header("💎 Криптовалюты")
        
        # Кнопки быстрого выбора
        col1, col2, col3 = st.sidebar.columns(3)
        with col1:
            if st.button("Топ-3", key="select_top3"):
                st.session_state.monitored_symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
        with col2:
            if st.button("Топ-6", key="select_top6"):
                st.session_state.monitored_symbols = self.available_symbols[:6]
        with col3:
            if st.button("Все 15", key="select_all"):
                st.session_state.monitored_symbols = self.available_symbols
        
        st.session_state.monitored_symbols = st.sidebar.multiselect(
            "Выберите монеты:",
            self.available_symbols,
            default=st.session_state.monitored_symbols,
            format_func=lambda x: self.symbol_names.get(x, x)
        )
        
        # Выбор стратегий
        st.sidebar.header("🤖 Стратегии")
        
        available_strategies = self.get_available_strategies()
        
        # Кнопки быстрого выбора стратегий
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("Базовые", key="select_basic"):
                st.session_state.selected_strategies = ['mean_reversion', 'trend_following', 'breakout']
        with col2:
            if st.button("Все", key="select_all_strategies"):
                st.session_state.selected_strategies = list(available_strategies.keys())
        
        selected_strategy_names = []
        for key, info in available_strategies.items():
            if st.sidebar.checkbox(info['name'], value=key in st.session_state.selected_strategies, key=f"strat_{key}"):
                if key not in selected_strategy_names:
                    selected_strategy_names.append(key)
        
        st.session_state.selected_strategies = selected_strategy_names
        
        # Статус активных позиций
        try:
            summary = self.position_tracker.get_position_summary()
            if summary['active_positions'] > 0:
                st.sidebar.header("📊 Ваши позиции")
                st.sidebar.success(f"🟢 Активных: {summary['active_positions']}")
                if summary['total_unrealized_pnl_pct'] != 0:
                    pnl_color = "🟢" if summary['total_unrealized_pnl_pct'] > 0 else "🔴"
                    st.sidebar.info(f"{pnl_color} P&L: {summary['total_unrealized_pnl_pct']:+.2f}%")
                if summary['alerts_count'] > 0:
                    st.sidebar.warning(f"🚨 Алертов: {summary['alerts_count']}")
        except Exception as e:
            st.sidebar.error(f"Ошибка позиций: {e}")
        
        # Статус мониторинга
        if st.session_state.monitoring_active:
            st.sidebar.success(f"🟢 Мониторинг: {len(st.session_state.monitored_symbols)} монет")
            st.sidebar.info(f"🎯 Фильтр: {st.session_state.min_signal_confidence}%+")
            
            # Показываем время до следующего обновления
            time_since_update = time.time() - st.session_state.last_update_time
            time_to_next = max(0, st.session_state.update_interval - time_since_update)
            if time_to_next > 0:
                st.sidebar.info(f"⏰ Следующее обновление через: {int(time_to_next)} сек")
        else:
            st.sidebar.warning("🔴 Мониторинг остановлен")
    
    def render_alerts_panel(self, analysis_results: List[Dict]):
        """Панель активных сигналов с рекомендациями - ИСПРАВЛЕННАЯ"""
        
        st.markdown("## 🚨 Активные сигналы")
        
        # Группируем сигналы по символам (один лучший сигнал на монету)
        grouped_signals = self.group_signals_by_symbol(analysis_results)
        
        if grouped_signals:
            # Показываем статистику фильтра
            st.info(f"🎯 Показаны сигналы с уверенностью {st.session_state.min_signal_confidence}%+ • Найдено: {len(grouped_signals)} монет")
            
            # Сортируем по уверенности
            sorted_signals = sorted(grouped_signals.values(), key=lambda x: x['confidence'], reverse=True)
            
            # Отображаем в колонках
            cols = st.columns(min(len(sorted_signals), 3))
            
            for i, alert in enumerate(sorted_signals[:9]):  # Максимум 9 алертов
                with cols[i % 3]:
                    
                    # Определяем класс карточки
                    if alert['strength'] == 'СИЛЬНЫЙ' and alert['action'] == 'BUY':
                        card_class = "signal-strong-buy"
                    elif alert['strength'] == 'СИЛЬНЫЙ' and alert['action'] == 'SELL':
                        card_class = "signal-strong-sell"
                    elif alert['action'] == 'BUY':
                        card_class = "signal-buy"
                    elif alert['action'] == 'SELL':
                        card_class = "signal-sell"
                    else:
                        card_class = "signal-hold"
                    
                    price_change = ((alert['current_price'] - alert['price']) / alert['price']) * 100
                    
                    # Получаем информацию о стратегии
                    strategy_info = self.get_available_strategies().get(alert['strategy'], {})
                    strategy_name = strategy_info.get('name', alert['strategy'])
                    
                    # ИСПРАВЛЕНО: Отображаем карточку сигнала БЕЗ HTML
                    st.markdown(f"""
                    <div class="signal-card {card_class}">
                        <h4>{alert['color']} {alert['symbol_name']}</h4>
                        <p><strong>{alert['action']}</strong> • {strategy_name}</p>
                        <p>💪 {alert['strength']} ({alert['confidence']:.0f}%)</p>
                        <p>💰 ${alert['current_price']:.4f} ({price_change:+.2f}%)</p>
                        <small>{alert['reason']}</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # ИСПРАВЛЕНО: Отображаем рекомендации ОТДЕЛЬНО через st.info()
                    if alert.get('recommended_stop_loss') and alert.get('recommended_take_profit'):
                        st.info(f"""
🛡️ **Стоп:** ${alert['recommended_stop_loss']:.4f} (-{alert['stop_pct']:.1f}%)  
🎯 **Профит:** ${alert['recommended_take_profit']:.4f} (+{alert['tp_pct']:.1f}%)  
⚖️ **R/R:** 1:{alert['risk_reward_ratio']:.1f}
""")
                    
                    # ИСПРАВЛЕНО: Кнопка быстрого входа в позицию
                    if st.button(f"📈 Войти в позицию", key=f"enter_{alert['symbol']}_{i}", use_container_width=True):
                        # Сохраняем данные сигнала для быстрого входа
                        st.session_state.quick_entry_data = {
                            'symbol': alert['symbol'],
                            'direction': 'LONG' if alert['action'] == 'BUY' else 'SHORT',
                            'price': alert['current_price'],
                            'strategy': alert['strategy'],
                            'confidence': alert['confidence'],
                            'recommended_stop_loss': alert.get('recommended_stop_loss'),
                            'recommended_take_profit': alert.get('recommended_take_profit'),
                            'symbol_name': alert['symbol_name']
                        }
                        st.success("✅ Данные сохранены! Перейдите на вкладку 'Позиции' для завершения входа.")
                        
        else:
            st.info(f"🔭 Нет сигналов с уверенностью {st.session_state.min_signal_confidence}%+ за последние 6 часов. Попробуйте снизить фильтр или добавить больше монет/стратегий.")
    
    def render_overview_table(self, analysis_results: List[Dict]):
        """Обзорная таблица всех монет"""
        
        st.markdown("## 📊 Обзор рынка")
        
        # Подготавливаем данные для таблицы
        table_data = []
        
        for result in analysis_results:
            if 'error' not in result:
                symbol_name = self.symbol_names.get(result['symbol'], result['symbol'])
                
                # Консенсус
                consensus_emoji = {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '🟡'}
                consensus_text = f"{consensus_emoji[result['consensus']]} {result['consensus']}"
                
                # Рыночные условия
                market = result.get('market_conditions', {})
                trend = market.get('trend', 'N/A')
                volatility = market.get('volatility', 'N/A')
                
                table_data.append({
                    'Криптовалюта': symbol_name,
                    'Цена': f"${result['current_price']:.4f}",
                    'Изменение 24ч': f"{result['price_change_24h']:+.2f}%",
                    'Тренд': trend,
                    'Волатильность': volatility,
                    'Консенсус': consensus_text,
                    'Сила сигнала': f"{result['consensus_strength']:.0f}%",
                    'Активных сигналов': result['active_signals'],
                    'Buy/Sell': f"{result['buy_signals']}/{result['sell_signals']}"
                })
        
        if table_data:
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Краткая статистика
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_change = np.mean([r['price_change_24h'] for r in analysis_results if 'error' not in r])
                st.metric("Средний рост 24ч", f"{avg_change:+.2f}%")
            
            with col2:
                total_buy = sum(r['buy_signals'] for r in analysis_results if 'error' not in r)
                st.metric("Всего BUY сигналов", total_buy)
            
            with col3:
                total_sell = sum(r['sell_signals'] for r in analysis_results if 'error' not in r)
                st.metric("Всего SELL сигналов", total_sell)
            
            with col4:
                total_active = sum(r['active_signals'] for r in analysis_results if 'error' not in r)
                st.metric("Всего активных", total_active)
    
    def render_positions_tab(self):
        """Вкладка управления позициями"""
        
        st.markdown("## 📊 Управление позициями")
        
        # Обновляем цены для всех активных позиций
        active_positions = self.position_tracker.get_active_positions()
        if active_positions:
            # Получаем текущие цены
            current_prices = {}
            for position in active_positions:
                symbol_clean = position.symbol.replace('/', '')
                try:
                    price = self.data_provider.get_current_price(symbol_clean)
                    if price > 0:
                        current_prices[symbol_clean] = price
                except:
                    pass
            
            # Обновляем позиции
            if current_prices:
                self.position_tracker.update_position_prices(current_prices)
        
        # Сводка по позициям
        try:
            summary = self.position_tracker.get_position_summary()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Активных позиций", summary['active_positions'])
            with col2:
                st.metric("Закрытых позиций", summary['closed_positions'])
            with col3:
                pnl_color = "normal"
                if summary['total_unrealized_pnl_pct'] > 0:
                    pnl_color = "inverse"
                st.metric("Общий P&L", f"{summary['total_unrealized_pnl_pct']:+.2f}%", delta_color=pnl_color)
            with col4:
                st.metric("Алертов", summary['alerts_count'])
        except Exception as e:
            st.error(f"Ошибка получения сводки позиций: {e}")
        
        # Вкладки для позиций
        pos_tab1, pos_tab2, pos_tab3 = st.tabs(["➕ Новая позиция", "📈 Активные позиции", "📊 История"])
        
        with pos_tab1:
            self.render_add_position_form()
        
        with pos_tab2:
            self.render_active_positions()
        
        with pos_tab3:
            self.render_positions_history()
    
    def render_add_position_form(self):
        """Форма добавления новой позиции - ИСПРАВЛЕННАЯ"""
        
        st.markdown("### ➕ Добавить новую позицию")
        
        # ИСПРАВЛЕНО: Проверяем быстрый вход из сигнала
        if st.session_state.quick_entry_data is not None:
            quick = st.session_state.quick_entry_data
            st.success(f"🚀 Быстрый вход: {quick['symbol_name']} • {quick['direction']} • ${quick['price']:.4f}")
            
            # Показываем рекомендации из сигнала
            if quick.get('recommended_stop_loss') and quick.get('recommended_take_profit'):
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"🛡️ Рекомендуемый стоп: ${quick['recommended_stop_loss']:.4f}")
                with col2:
                    st.info(f"🎯 Рекомендуемый профит: ${quick['recommended_take_profit']:.4f}")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Подтвердить быстрый вход", type="primary", use_container_width=True):
                    try:
                        position_id = self.position_tracker.add_position(
                            symbol=quick['symbol'],
                            direction=quick['direction'],
                            entry_price=quick['price'],
                            quantity=0.01,  # Минимальный размер для демо
                            strategy=quick['strategy'],
                            confidence=quick['confidence'],
                            custom_stop_loss=quick.get('recommended_stop_loss'),
                            custom_take_profit=quick.get('recommended_take_profit')
                        )
                        
                        st.success(f"✅ Позиция {position_id} добавлена!")
                        st.session_state.quick_entry_data = None  # Очищаем данные
                        st.balloons()
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Ошибка: {e}")
            
            with col2:
                if st.button("❌ Отменить быстрый вход", use_container_width=True):
                    st.session_state.quick_entry_data = None  # Очищаем данные
                    st.rerun()
            
            st.divider()
        
        # Обычная форма добавления позиции
        col1, col2 = st.columns(2)
        
        with col1:
            symbol = st.selectbox(
                "Выберите криптовалюту:",
                self.available_symbols,
                format_func=lambda x: self.symbol_names.get(x, x),
                key="new_position_symbol"
            )
            
            direction = st.selectbox(
                "Направление позиции:",
                ["LONG", "SHORT"],
                key="new_position_direction"
            )
            
            entry_price = st.number_input(
                "Цена входа:",
                min_value=0.0001,
                step=0.0001,
                format="%.4f",
                key="new_position_price"
            )
        
        with col2:
            # Получаем текущую цену для подсказки
            if symbol:
                try:
                    current_price = self.data_provider.get_current_price(symbol.replace('/', ''))
                    if current_price > 0:
                        st.info(f"💰 Текущая цена: ${current_price:.4f}")
                        if entry_price == 0:
                            entry_price = current_price
                except:
                    pass
            
            quantity = st.number_input(
                "Размер позиции:",
                min_value=0.0001,
                step=0.0001,
                format="%.4f",
                key="new_position_quantity"
            )
            
            strategy = st.selectbox(
                "Стратегия:",
                ["manual"] + list(self.get_available_strategies().keys()),
                format_func=lambda x: "🤙 Ручной вход" if x == "manual" else self.get_available_strategies()[x]['name'],
                key="new_position_strategy"
            )
        
        # Автоматические рекомендации по риск-менеджменту
        if entry_price > 0 and symbol and direction:
            
            st.markdown("### 🛡️ Рекомендации по риск-менеджменту")
            
            account_balance = st.number_input(
                "Баланс аккаунта ($):",
                min_value=100.0,
                value=10000.0,
                step=100.0,
                key="account_balance"
            )
            
            recommendations = self.position_tracker.get_risk_recommendations(
                symbol, entry_price, direction, account_balance
            )
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                use_recommended_sl = st.checkbox("Использовать рекомендуемый стоп-лосс", value=True)
                if use_recommended_sl:
                    stop_loss = recommendations['recommended_stop_loss']
                    st.info(f"🛡️ Стоп-лосс: ${stop_loss:.4f}")
                else:
                    stop_loss = st.number_input(
                        "Стоп-лосс:",
                        min_value=0.0001,
                        value=recommendations['recommended_stop_loss'],
                        step=0.0001,
                        format="%.4f",
                        key="custom_stop_loss"
                    )
            
            with col2:
                use_recommended_tp = st.checkbox("Использовать рекомендуемый тейк-профит", value=True)
                if use_recommended_tp:
                    take_profit = recommendations['recommended_take_profit']
                    st.info(f"🎯 Тейк-профит: ${take_profit:.4f}")
                else:
                    take_profit = st.number_input(
                        "Тейк-профит:",
                        min_value=0.0001,
                        value=recommendations['recommended_take_profit'],
                        step=0.0001,
                        format="%.4f",
                        key="custom_take_profit"
                    )
            
            with col3:
                use_recommended_size = st.checkbox("Использовать рекомендуемый размер", value=False)
                if use_recommended_size:
                    quantity = recommendations['position_size']
                    st.info(f"📊 Размер: {quantity:.4f}")
            
            # Показываем детальные рекомендации
            with st.expander("📋 Детальные рекомендации", expanded=False):
                for note in recommendations['notes']:
                    if note:
                        st.write(note)
            
            # Кнопка добавления позиции
            if st.button("✅ Добавить позицию", type="primary", use_container_width=True):
                if all([symbol, direction, entry_price > 0, quantity > 0]):
                    try:
                        position_id = self.position_tracker.add_position(
                            symbol=symbol,
                            direction=direction,
                            entry_price=entry_price,
                            quantity=quantity,
                            strategy=strategy,
                            confidence=85,  # Для ручных позиций
                            custom_stop_loss=stop_loss if not use_recommended_sl else None,
                            custom_take_profit=take_profit if not use_recommended_tp else None
                        )
                        
                        st.success(f"✅ Позиция {position_id} успешно добавлена!")
                        st.balloons()
                        
                        # Очищаем форму
                        for key in ['new_position_price', 'new_position_quantity']:
                            if key in st.session_state:
                                st.session_state[key] = 0.0
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Ошибка добавления позиции: {e}")
                else:
                    st.error("❌ Заполните все обязательные поля")
    
    def render_active_positions(self):
        """Отображение активных позиций"""
        
        try:
            active_positions = self.position_tracker.get_active_positions()
            
            if not active_positions:
                st.info("🔭 Нет активных позиций")
                return
            
            st.markdown("### 📈 Активные позиции")
            
            for position in active_positions:
                
                # Определяем цвет карточки по P&L
                pnl_pct = position.unrealized_pnl_pct or 0
                if pnl_pct > 0:
                    card_color = "#d4edda"  # Зеленый
                    border_color = "#28a745"
                elif pnl_pct < -2:
                    card_color = "#f8d7da"  # Красный
                    border_color = "#dc3545"
                else:
                    card_color = "#fff3cd"  # Желтый
                    border_color = "#ffc107"
                
                # Карточка позиции
                with st.container():
                    st.markdown(f"""
                    <div style="padding: 1rem; margin: 0.5rem 0; border-radius: 10px; 
                               background-color: {card_color}; border-left: 5px solid {border_color};">
                        <h4>{self.symbol_names.get(position.symbol, position.symbol)} • {position.direction}</h4>
                        <p><strong>ID:</strong> {position.id}</p>
                        <p><strong>Стратегия:</strong> {position.strategy}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Цена входа", f"${position.entry_price:.4f}")
                        st.metric("Размер", f"{position.quantity:.4f}")
                    
                    with col2:
                        st.metric("Текущая цена", f"${position.current_price:.4f}" if position.current_price else "N/A")
                        st.metric("P&L", f"{pnl_pct:+.2f}%" if position.unrealized_pnl_pct else "N/A")
                    
                    with col3:
                        st.metric("Стоп-лосс", f"${position.stop_loss:.4f}" if position.stop_loss else "N/A")
                        st.metric("Тейк-профит", f"${position.take_profit:.4f}" if position.take_profit else "N/A")
                    
                    with col4:
                        if position.trailing_stop:
                            st.metric("Трейлинг-стоп", f"${position.trailing_stop:.4f}")
                        
                        # Время в позиции
                        time_in_position = datetime.now() - position.entry_time
                        hours = int(time_in_position.total_seconds() / 3600)
                        st.metric("Время в позиции", f"{hours}ч")
                    
                    # Алерты по позиции
                    if position.exit_alerts:
                        st.markdown("**🚨 Алерты:**")
                        for alert in position.exit_alerts[-3:]:  # Показываем последние 3 алерта
                            st.warning(alert)
                    
                    # Кнопки управления
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button(f"❌ Закрыть позицию", key=f"close_{position.id}"):
                            # Получаем текущую цену для закрытия
                            current_price = position.current_price or position.entry_price
                            self.position_tracker.close_position(position.id, current_price, "Manual close")
                            st.success("✅ Позиция закрыта!")
                            st.rerun()
                    
                    with col2:
                        if st.button(f"🔄 Обновить цену", key=f"update_{position.id}"):
                            try:
                                symbol_clean = position.symbol.replace('/', '')
                                new_price = self.data_provider.get_current_price(symbol_clean)
                                if new_price > 0:
                                    self.position_tracker.update_position_prices({symbol_clean: new_price})
                                    st.success("✅ Цена обновлена!")
                                    st.rerun()
                            except Exception as e:
                                st.error(f"❌ Ошибка обновления: {e}")
                    
                    with col3:
                        if st.button(f"📊 Детали", key=f"details_{position.id}"):
                            st.session_state[f"show_details_{position.id}"] = True
                    
                    st.divider()
        except Exception as e:
            st.error(f"Ошибка отображения активных позиций: {e}")
    
    def render_positions_history(self):
        """История закрытых позиций"""
        
        try:
            real_trades = self.position_tracker.get_real_trade_history()
            
            if not real_trades:
                st.info("🔭 История реальных сделок пуста")
                return
            
            st.markdown("### 📊 История реальных сделок")
            
            # Создаем DataFrame для отображения
            df_real_history = pd.DataFrame(real_trades)
            
            # Форматируем для отображения
            df_display = df_real_history.copy()
            df_display['timestamp'] = df_display['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            df_display['symbol'] = df_display['symbol'].apply(lambda x: self.symbol_names.get(x, x))
            df_display['entry_price'] = df_display['entry_price'].apply(lambda x: f"${x:.4f}")
            df_display['exit_price'] = df_display['exit_price'].apply(lambda x: f"${x:.4f}" if x else "Активна")
            df_display['pnl_pct'] = df_display['pnl_pct'].apply(lambda x: f"{x:+.2f}%" if x is not None else "N/A")
            
            # Переименовываем колонки
            df_display = df_display.rename(columns={
                'timestamp': 'Дата входа',
                'symbol': 'Криптовалюта', 
                'direction': 'Направление',
                'entry_price': 'Вход',
                'exit_price': 'Выход',
                'pnl_pct': 'P&L %',
                'strategy': 'Стратегия',
                'duration': 'Дней'
            })
            
            # Отображаем таблицу
            st.dataframe(df_display, use_container_width=True, hide_index=True)
            
            # Статистика по реальным сделкам
            closed_trades = [t for t in real_trades if t['pnl_pct'] is not None]
            
            if closed_trades:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_trades = len(closed_trades)
                    st.metric("Всего сделок", total_trades)
                
                with col2:
                    profitable_trades = len([t for t in closed_trades if t['pnl_pct'] > 0])
                    win_rate = profitable_trades / total_trades * 100 if total_trades > 0 else 0
                    st.metric("Винрейт", f"{win_rate:.1f}%")
                
                with col3:
                    avg_profit = np.mean([t['pnl_pct'] for t in closed_trades if t['pnl_pct'] > 0]) if any(t['pnl_pct'] > 0 for t in closed_trades) else 0
                    st.metric("Средняя прибыль", f"{avg_profit:.2f}%")
                
                with col4:
                    best_trade = max(t['pnl_pct'] for t in closed_trades) if closed_trades else 0
                    st.metric("Лучшая сделка", f"{best_trade:.2f}%")
            
            # Кнопка очистки истории реальных сделок
            if st.button("🗑️ Очистить историю сделок", key="clear_real_history"):
                # Удаляем только закрытые позиции
                active_positions = {k: v for k, v in self.position_tracker.positions.items() if v.is_active}
                self.position_tracker.positions = active_positions
                self.position_tracker.save_positions()
                st.success("История реальных сделок очищена!")
                st.rerun()
        except Exception as e:
            st.error(f"Ошибка отображения истории: {e}")
    
    def render_detailed_analysis(self):
        """Детальный анализ одной монеты"""
        
        st.markdown("## 📈 Детальный анализ")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            selected_symbol = st.selectbox(
                "Выберите криптовалюту:",
                self.available_symbols,
                format_func=lambda x: self.symbol_names.get(x, x)
            )
        
        with col2:
            timeframe = st.selectbox("Таймфрейм:", ['5m', '15m', '1h', '4h'])
        
        with col3:
            if st.button("🔍 Анализировать", use_container_width=True, key="analyze_button"):
                st.session_state.force_analysis = True
        
        # Выполняем анализ
        if 'force_analysis' in st.session_state and st.session_state.force_analysis:
            st.session_state.force_analysis = False
            
            with st.spinner(f"Анализ {self.symbol_names.get(selected_symbol, selected_symbol)}..."):
                result = self.analyze_symbol_enhanced(selected_symbol, timeframe)
                
                if 'error' not in result:
                    # Основные метрики
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Текущая цена",
                            f"${result['current_price']:.4f}",
                            f"{result['price_change_24h']:+.2f}%"
                        )
                    
                    with col2:
                        consensus_colors = {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '🟡'}
                        st.metric(
                            "Консенсус",
                            f"{consensus_colors[result['consensus']]} {result['consensus']}"
                        )
                    
                    with col3:
                        st.metric(
                            "Сила сигнала",
                            f"{result['consensus_strength']:.0f}%"
                        )
                    
                    with col4:
                        st.metric(
                            "Активных сигналов",
                            result['active_signals']
                        )
                    
                    # График цены с индикаторами
                    if not result['data'].empty:
                        fig = make_subplots(
                            rows=3, cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.05,
                            subplot_titles=('Цена', 'RSI', 'Объем'),
                            row_heights=[0.5, 0.25, 0.25]
                        )
                        
                        # Свечной график
                        fig.add_trace(
                            go.Candlestick(
                                x=result['data'].index,
                                open=result['data']['open'],
                                high=result['data']['high'],
                                low=result['data']['low'],
                                close=result['data']['close'],
                                name='Price'
                            ),
                            row=1, col=1
                        )
                        
                        # RSI
                        rsi = self.indicators.rsi(result['data']['close'])
                        fig.add_trace(
                            go.Scatter(x=result['data'].index, y=rsi, name='RSI'),
                            row=2, col=1
                        )
                        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                        
                        # Объем
                        fig.add_trace(
                            go.Bar(x=result['data'].index, y=result['data']['volume'], name='Volume'),
                            row=3, col=1
                        )
                        
                        fig.update_layout(height=700, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Сигналы по стратегиям
                    if result['signals']:
                        st.markdown("### 📊 Последние сигналы с рекомендациями")
                        
                        for signal in result['signals'][:5]:  # Показываем топ-5 сигналов
                            with st.expander(f"{signal['type']} • {signal.get('strategy', 'Unknown')} • {signal.get('confidence', 0):.0f}%"):
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.write(f"**Цена сигнала:** ${signal['price']:.4f}")
                                    st.write(f"**Время:** {signal['timestamp']}")
                                    st.write(f"**Причина:** {signal.get('reason', 'N/A')}")
                                
                                with col2:
                                    if signal.get('recommended_stop_loss'):
                                        st.write(f"**🛡️ Стоп-лосс:** ${signal['recommended_stop_loss']:.4f}")
                                        st.write(f"**Риск:** -{signal.get('stop_pct', 0):.1f}%")
                                
                                with col3:
                                    if signal.get('recommended_take_profit'):
                                        st.write(f"**🎯 Тейк-профит:** ${signal['recommended_take_profit']:.4f}")
                                        st.write(f"**Потенциал:** +{signal.get('tp_pct', 0):.1f}%")
                                        st.write(f"**R/R:** 1:{signal.get('risk_reward_ratio', 0):.1f}")
                    
                    # Рыночные условия
                    st.markdown("### 🌡️ Рыночные условия")
                    market = result.get('market_conditions', {})
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Тренд", market.get('trend', 'N/A'))
                    with col2:
                        st.metric("Волатильность", market.get('volatility', 'N/A'))
                    with col3:
                        st.metric("Объем", market.get('volume', 'N/A'))
                    with col4:
                        recommended = market.get('recommended_strategies', [])
                        st.metric("Рекомендуемые стратегии", len(recommended))
                    
                    if recommended:
                        st.info(f"💡 Рекомендуемые стратегии: {', '.join(recommended)}")
                
                else:
                    st.error(result['error'])
    
    def render_strategies_info(self):
        """Информация о торговых стратегиях"""
        
        st.markdown("## ℹ️ О торговых стратегиях")
        
        strategies = self.get_available_strategies()
        
        for key, strategy in strategies.items():
            with st.expander(f"{strategy['name']}", expanded=False):
                st.markdown(f"**Описание:** {strategy['description']}")
                st.markdown(f"**Детали:** {strategy['details']}")
                
                # Дополнительная информация для каждой стратегии
                if key == 'mean_reversion':
                    st.markdown("""
                    **Параметры риск-менеджмента:**
                    - Стоп-лосс: 2.5%
                    - Тейк-профит: 5.0%
                    - Соотношение риск/доходность: 1:2
                    
                    **Лучше всего работает:**
                    - В боковых трендах
                    - При высокой волатильности
                    - На коротких таймфреймах
                    """)
                
                elif key == 'trend_following':
                    st.markdown("""
                    **Параметры риск-менеджмента:**
                    - Стоп-лосс: 3.0%
                    - Тейк-профит: 9.0%
                    - Соотношение риск/доходность: 1:3
                    
                    **Лучше всего работает:**
                    - В сильных трендах
                    - На средних и долгих таймфреймах
                    - При высоких объемах торгов
                    """)
                
                elif key == 'breakout':
                    st.markdown("""
                    **Параметры риск-менеджмента:**
                    - Стоп-лосс: 2.0%
                    - Тейк-профит: 6.0%
                    - Соотношение риск/доходность: 1:3
                    
                    **Лучше всего работает:**
                    - При пробоях уровней поддержки/сопротивления
                    - В периоды низкой волатильности с последующим всплеском
                    - При важных новостных событиях
                    """)
                
                elif key == 'scalping':
                    st.markdown("""
                    **Параметры риск-менеджмента:**
                    - Стоп-лосс: 1.5%
                    - Тейк-профит: 3.0%
                    - Соотношение риск/доходность: 1:2
                    
                    **Лучше всего работает:**
                    - На очень коротких таймфреймах (1m-5m)
                    - При высокой ликвидности
                    - Требует постоянного мониторинга
                    """)
                
                elif key == 'momentum':
                    st.markdown("""
                    **Параметры риск-менеджмента:**
                    - Стоп-лосс: 3.5%
                    - Тейк-профит: 7.0%
                    - Соотношение риск/доходность: 1:2
                    
                    **Лучше всего работает:**
                    - При сильных импульсных движениях
                    - После важных новостей
                    - При резких изменениях объемов
                    """)
        
        st.markdown("### 💡 Общие рекомендации")
        st.info("""
        - Используйте несколько стратегий одновременно для диверсификации
        - Всегда соблюдайте рекомендуемые уровни стоп-лосса и тейк-профита
        - Учитывайте рыночные условия при выборе стратегий
        - Тестируйте стратегии на исторических данных перед реальной торговлей
        """)
    
    def run(self):
        """Главная функция запуска дашборда - ИСПРАВЛЕННАЯ"""
        
        # Заголовок
        st.markdown("""
        <div class="main-header">
            <h1>🚀 Trading Bot Pro Dashboard</h1>
            <p>15 криптовалют • 5 продвинутых стратегий • Трекер позиций • Рекомендации стоп/профит</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Боковая панель
        self.render_enhanced_controls()
        
        # Основные вкладки
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🚨 Мониторинг", "📊 Позиции", "📈 Детальный анализ", "📊 История", "ℹ️ О стратегиях"])
        
        with tab1:
            if st.session_state.monitoring_active and st.session_state.monitored_symbols:
                
                # Проверяем нужно ли обновление - ИСПРАВЛЕННАЯ ЛОГИКА
                current_time = time.time()
                time_since_update = current_time - st.session_state.last_update_time
                
                if time_since_update >= st.session_state.update_interval:
                    # Основной цикл мониторинга
                    analysis_results = []
                    
                    with st.spinner("🔄 Анализ рынка..."):
                        progress_bar = st.progress(0)
                        
                        for i, symbol in enumerate(st.session_state.monitored_symbols):
                            result = self.analyze_symbol_enhanced(symbol)
                            analysis_results.append(result)
                            progress_bar.progress((i + 1) / len(st.session_state.monitored_symbols))
                        
                        progress_bar.empty()
                    
                    # Сохраняем результаты и время обновления
                    st.session_state.analysis_results = analysis_results
                    st.session_state.last_update_time = current_time
                else:
                    # Используем сохраненные результаты
                    analysis_results = st.session_state.get('analysis_results', [])
                
                if analysis_results:
                    # Отображение результатов
                    self.render_alerts_panel(analysis_results)
                    self.render_overview_table(analysis_results)
                
                # ИСПРАВЛЕННОЕ АВТООБНОВЛЕНИЕ - 30 секунд вместо 2
                if st.session_state.monitoring_active:
                    time_to_next = max(0, st.session_state.update_interval - time_since_update)
                    if time_to_next <= 5:  # Обновляем только если до следующего обновления меньше 5 секунд
                        time.sleep(5)
                        st.rerun()
                    else:
                        # Показываем обратный отсчет без перезагрузки
                        st.info(f"⏰ Следующее обновление через: {int(time_to_next)} секунд")
            
            else:
                st.info("👆 Запустите мониторинг в боковой панели для начала анализа")
                
                # Показываем возможности
                st.markdown("### 🎯 Возможности системы:")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    **💎 15 Топовых криптовалют:**
                    - Bitcoin, Ethereum, Binance Coin
                    - Solana, Cardano, Litecoin
                    - Avalanche, Polkadot, Chainlink
                    - Ripple, Dogecoin, Uniswap
                    - Bitcoin Cash, Cosmos, Algorand
                    """)
                
                with col2:
                    st.markdown("""
                    **🛡️ Умные рекомендации:**
                    - Автоматический расчет стоп-лосса
                    - Рекомендации по тейк-профиту
                    - Соотношение риск/доходность
                    - Быстрый вход в позицию одним кликом
                    """)
        
        with tab2:
            self.render_positions_tab()
        
        with tab3:
            self.render_detailed_analysis()
        
        with tab4:
            st.markdown("## 📊 История реальных сделок")
            
            # Получаем историю только реальных позиций
            try:
                real_trades = self.position_tracker.get_real_trade_history()
                
                if real_trades:
                    st.markdown("### 📈 Ваши сделки")
                    
                    # Создаем DataFrame из реальных сделок
                    df_real_history = pd.DataFrame(real_trades)
                    
                    # Форматируем для отображения
                    df_display = df_real_history.copy()
                    df_display['timestamp'] = df_display['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
                    df_display['symbol'] = df_display['symbol'].apply(lambda x: self.symbol_names.get(x, x))
                    df_display['entry_price'] = df_display['entry_price'].apply(lambda x: f"${x:.4f}")
                    df_display['exit_price'] = df_display['exit_price'].apply(lambda x: f"${x:.4f}" if x else "Активна")
                    df_display['pnl_pct'] = df_display['pnl_pct'].apply(lambda x: f"{x:+.2f}%" if x is not None else "N/A")
                    
                    # Переименовываем колонки
                    df_display = df_display.rename(columns={
                        'timestamp': 'Дата входа',
                        'symbol': 'Криптовалюта', 
                        'direction': 'Направление',
                        'entry_price': 'Вход',
                        'exit_price': 'Выход',
                        'pnl_pct': 'P&L %',
                        'strategy': 'Стратегия',
                        'duration': 'Дней'
                    })
                    
                    # Отображаем таблицу
                    st.dataframe(df_display, use_container_width=True, hide_index=True)
                    
                    # Статистика по реальным сделкам
                    closed_trades = [t for t in real_trades if t['pnl_pct'] is not None]
                    
                    if closed_trades:
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            total_trades = len(closed_trades)
                            st.metric("Всего сделок", total_trades)
                        
                        with col2:
                            profitable_trades = len([t for t in closed_trades if t['pnl_pct'] > 0])
                            win_rate = profitable_trades / total_trades * 100 if total_trades > 0 else 0
                            st.metric("Винрейт", f"{win_rate:.1f}%")
                        
                        with col3:
                            avg_profit = np.mean([t['pnl_pct'] for t in closed_trades if t['pnl_pct'] > 0]) if any(t['pnl_pct'] > 0 for t in closed_trades) else 0
                            st.metric("Средняя прибыль", f"{avg_profit:.2f}%")
                        
                        with col4:
                            best_trade = max(t['pnl_pct'] for t in closed_trades) if closed_trades else 0
                            st.metric("Лучшая сделка", f"{best_trade:.2f}%")
                    
                    # График статистики сделок
                    if len(closed_trades) > 5:
                        st.markdown("### 📈 График P&L сделок")
                        
                        # Создаем график P&L по времени
                        pnl_data = []
                        cumulative_pnl = 0
                        
                        for trade in sorted(closed_trades, key=lambda x: x['timestamp']):
                            cumulative_pnl += trade['pnl_pct']
                            pnl_data.append({
                                'Date': trade['timestamp'],
                                'Individual P&L': trade['pnl_pct'],
                                'Cumulative P&L': cumulative_pnl
                            })
                        
                        pnl_df = pd.DataFrame(pnl_data)
                        
                        fig = make_subplots(
                            rows=2, cols=1,
                            subplot_titles=('Отдельные сделки', 'Накопительный P&L'),
                            vertical_spacing=0.1
                        )
                        
                        # График отдельных сделок
                        colors = ['green' if x > 0 else 'red' for x in pnl_df['Individual P&L']]
                        fig.add_trace(
                            go.Bar(
                                x=pnl_df['Date'],
                                y=pnl_df['Individual P&L'],
                                name='P&L сделки',
                                marker_color=colors
                            ),
                            row=1, col=1
                        )
                        
                        # График накопительного P&L
                        fig.add_trace(
                            go.Scatter(
                                x=pnl_df['Date'],
                                y=pnl_df['Cumulative P&L'],
                                mode='lines+markers',
                                name='Накопительный P&L',
                                line=dict(color='blue', width=2)
                            ),
                            row=2, col=1
                        )
                        
                        fig.update_layout(height=600, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.info("🔭 История реальных сделок пуста. Начните торговать для накопления статистики!")
                    
                    # Показываем пример того, что будет доступно
                    st.markdown("### 📊 Доступная статистика после первых сделок:")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("""
                        **📈 Базовая статистика:**
                        - Общее количество сделок
                        - Процент прибыльных сделок (винрейт)
                        - Средняя прибыль/убыток
                        """)
                    
                    with col2:
                        st.markdown("""
                        **📊 Графики и аналитика:**
                        - График P&L по времени
                        - Накопительная доходность
                        - Анализ по стратегиям
                        """)
                    
                    with col3:
                        st.markdown("""
                        **🎯 Рекомендации:**
                        - Лучшие и худшие сделки
                        - Анализ временных интервалов
                        - Оптимизация стратегий
                        """)
            
            except Exception as e:
                st.error(f"Ошибка загрузки истории: {e}")
        
        with tab5:
            self.render_strategies_info()


# Главная функция запуска приложения
def main():
    """Главная функция приложения"""
    try:
        dashboard = EnhancedTradingDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"❌ Критическая ошибка приложения: {e}")
        st.markdown("""
        ### 🔧 Возможные решения:
        1. **Перезапустите приложение** - `streamlit run app.py`
        2. **Проверьте все модули** - убедитесь что все файлы на месте
        3. **Очистите кэш** - `streamlit cache clear`
        4. **Проверьте интернет** - для загрузки данных
        
        ### 📋 Список необходимых файлов:
        - `data_provider.py`
        - `strategies.py` 
        - `indicators.py`
        - `dominance_strategy.py`
        - `backtester.py`
        - `position_tracker.py`
        """)


if __name__ == "__main__":
    main()