import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class Trade:
    """Класс для представления сделки"""
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    direction: str  # 'LONG' или 'SHORT'
    quantity: float
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    exit_reason: str = "OPEN"
    duration_hours: Optional[float] = None
    strategy: str = ""

class Position:
    """Класс для управления позицией"""
    
    def __init__(self):
        self.is_open = False
        self.direction = None  # 'LONG' или 'SHORT'
        self.entry_price = 0
        self.entry_time = None
        self.quantity = 0
        self.stop_loss = None
        self.take_profit = None
        self.strategy = ""
    
    def open_position(self, direction: str, price: float, quantity: float, 
                     timestamp: datetime, stop_loss: float = None, 
                     take_profit: float = None, strategy: str = ""):
        """Открытие позиции"""
        self.is_open = True
        self.direction = direction
        self.entry_price = price
        self.entry_time = timestamp
        self.quantity = quantity
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.strategy = strategy
    
    def close_position(self, price: float, timestamp: datetime, reason: str = "SIGNAL") -> Trade:
        """Закрытие позиции и возврат объекта Trade"""
        if not self.is_open:
            return None
        
        # Расчет P&L
        if self.direction == 'LONG':
            pnl = (price - self.entry_price) * self.quantity
            pnl_pct = (price / self.entry_price - 1) * 100
        else:  # SHORT
            pnl = (self.entry_price - price) * self.quantity
            pnl_pct = (self.entry_price / price - 1) * 100
        
        # Длительность сделки
        duration = (timestamp - self.entry_time).total_seconds() / 3600
        
        trade = Trade(
            entry_time=self.entry_time,
            exit_time=timestamp,
            entry_price=self.entry_price,
            exit_price=price,
            direction=self.direction,
            quantity=self.quantity,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            duration_hours=duration,
            strategy=self.strategy
        )
        
        # Сброс позиции
        self.is_open = False
        self.direction = None
        self.entry_price = 0
        self.entry_time = None
        self.quantity = 0
        self.stop_loss = None
        self.take_profit = None
        self.strategy = ""
        
        return trade
    
    def check_exit_conditions(self, current_price: float) -> Tuple[bool, str]:
        """Проверка условий выхода из позиции"""
        if not self.is_open:
            return False, ""
        
        # Проверка стоп-лосса
        if self.stop_loss:
            if self.direction == 'LONG' and current_price <= self.stop_loss:
                return True, "STOP_LOSS"
            elif self.direction == 'SHORT' and current_price >= self.stop_loss:
                return True, "STOP_LOSS"
        
        # Проверка тейк-профита
        if self.take_profit:
            if self.direction == 'LONG' and current_price >= self.take_profit:
                return True, "TAKE_PROFIT"
            elif self.direction == 'SHORT' and current_price <= self.take_profit:
                return True, "TAKE_PROFIT"
        
        return False, ""

class Backtester:
    """Класс для бэктестирования торговых стратегий"""
    
    def __init__(self, initial_balance: float = 10000, commission: float = 0.001, 
                 slippage: float = 0.0005, max_positions: int = 1, risk_per_trade: float = 0.02):
        """
        Инициализация бэктестера
        
        Args:
            initial_balance: Начальный капитал
            commission: Комиссия за сделку (0.001 = 0.1%)
            slippage: Проскальзывание цены (0.0005 = 0.05%)
            max_positions: Максимальное количество одновременных позиций
            risk_per_trade: Риск на сделку в долях от капитала (0.02 = 2%)
        """
        self.initial_balance = initial_balance
        self.commission = commission
        self.slippage = slippage
        self.max_positions = max_positions
        self.risk_per_trade = risk_per_trade
        
        # Состояние счета
        self.balance = initial_balance
        self.equity = initial_balance
        self.peak_balance = initial_balance
        
        # Позиции и сделки
        self.positions = []
        self.completed_trades = []
        self.equity_curve = []
        
        # Статистика
        self.trade_count = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
    def reset(self):
        """Сброс состояния бэктестера"""
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.peak_balance = self.initial_balance
        self.positions = []
        self.completed_trades = []
        self.equity_curve = []
        self.trade_count = 0
        self.winning_trades = 0
        self.losing_trades = 0
    
    def calculate_position_size(self, entry_price: float, stop_loss: float = None) -> float:
        """Расчет размера позиции на основе риск-менеджмента"""
        if stop_loss is None:
            # Если нет стоп-лосса, используем фиксированный процент от баланса
            return self.balance * self.risk_per_trade / entry_price
        
        # Расчет размера позиции по методу фиксированного риска
        risk_amount = self.balance * self.risk_per_trade
        price_risk = abs(entry_price - stop_loss)
        
        if price_risk == 0:
            return self.balance * self.risk_per_trade / entry_price
        
        position_size = risk_amount / price_risk
        
        # Ограничение максимального размера позиции
        max_position_value = self.balance * 0.95  # Максимум 95% от баланса
        max_quantity = max_position_value / entry_price
        
        return min(position_size, max_quantity)
    
    def apply_costs(self, quantity: float, price: float) -> float:
        """Применение комиссий и проскальзывания"""
        position_value = quantity * price
        commission_cost = position_value * self.commission
        slippage_cost = position_value * self.slippage
        return commission_cost + slippage_cost
    
    def process_signal(self, signal: Dict, current_price: float, timestamp: datetime):
        """Обработка торгового сигнала"""
        signal_type = signal.get('type', signal.get('action', 'HOLD'))
        
        if signal_type in ['BUY', 'SELL']:
            # Проверяем, есть ли свободное место для новой позиции
            active_positions = [p for p in self.positions if p.is_open]
            
            if len(active_positions) >= self.max_positions:
                return  # Максимальное количество позиций уже открыто
            
            # Определяем направление
            direction = 'LONG' if signal_type == 'BUY' else 'SHORT'
            
            # Получаем уровни стоп-лосса и тейк-профита
            stop_loss = signal.get('stop_loss')
            take_profit = signal.get('take_profit')
            strategy_name = signal.get('strategy', 'Unknown')
            
            # Расчет размера позиции
            quantity = self.calculate_position_size(current_price, stop_loss)
            
            if quantity <= 0:
                return
            
            # Применение затрат на открытие позиции
            costs = self.apply_costs(quantity, current_price)
            
            # Проверяем, достаточно ли средств
            required_margin = quantity * current_price + costs
            if required_margin > self.balance:
                return
            
            # Открываем позицию
            position = Position()
            position.open_position(
                direction=direction,
                price=current_price,
                quantity=quantity,
                timestamp=timestamp,
                stop_loss=stop_loss,
                take_profit=take_profit,
                strategy=strategy_name
            )
            
            self.positions.append(position)
            
            # Обновляем баланс
            self.balance -= costs
            self.trade_count += 1
    
    def update_equity(self, current_price: float, timestamp: datetime):
        """Обновление текущего капитала"""
        # Базовый баланс
        equity = self.balance
        
        # Добавляем нереализованную прибыль/убыток от открытых позиций
        for position in self.positions:
            if position.is_open:
                if position.direction == 'LONG':
                    unrealized_pnl = (current_price - position.entry_price) * position.quantity
                else:  # SHORT
                    unrealized_pnl = (position.entry_price - current_price) * position.quantity
                
                equity += unrealized_pnl
        
        self.equity = equity
        
        # Обновляем пиковый баланс
        if self.equity > self.peak_balance:
            self.peak_balance = self.equity
        
        # Сохраняем точку кривой капитала
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': self.equity,
            'balance': self.balance,
            'drawdown': (self.peak_balance - self.equity) / self.peak_balance * 100
        })
    
    def check_exit_conditions(self, current_price: float, timestamp: datetime):
        """Проверка условий выхода для всех открытых позиций"""
        positions_to_close = []
        
        for i, position in enumerate(self.positions):
            if position.is_open:
                should_exit, exit_reason = position.check_exit_conditions(current_price)
                
                if should_exit:
                    positions_to_close.append((i, exit_reason))
        
        # Закрываем позиции
        for i, exit_reason in reversed(positions_to_close):  # Обратный порядок для корректного удаления
            position = self.positions[i]
            trade = position.close_position(current_price, timestamp, exit_reason)
            
            if trade:
                # Применяем затраты на закрытие
                costs = self.apply_costs(trade.quantity, current_price)
                net_pnl = trade.pnl - costs
                
                # Обновляем баланс
                self.balance += net_pnl + (trade.quantity * trade.entry_price)  # Возвращаем маржу
                
                # Обновляем статистику
                if net_pnl > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1
                
                # Сохраняем сделку с учетом затрат
                trade.pnl = net_pnl
                if trade.entry_price != 0:
                    trade.pnl_pct = (net_pnl / (trade.quantity * trade.entry_price)) * 100
                
                self.completed_trades.append(trade)
    
    def run_backtest(self, data: pd.DataFrame, strategy, verbose: bool = True) -> Dict:
        """
        Запуск бэктестирования
        
        Args:
            data: DataFrame с OHLCV данными
            strategy: Объект стратегии или функция генерации сигналов
            verbose: Выводить ли прогресс
            
        Returns:
            Dict с результатами бэктестирования
        """
        try:
            self.reset()
            
            # Генерируем сигналы
            if hasattr(strategy, 'generate_signals'):
                # Объект стратегии
                signals = strategy.generate_signals(data)
                strategy_name = strategy.name
            elif hasattr(strategy, 'combine_signals'):
                # TradingStrategies объект
                signals = strategy.combine_signals(data)
                strategy_name = "Combined_Strategies"
            else:
                return {'error': 'Неподдерживаемый тип стратегии'}
            
            if not signals:
                return {'error': 'Стратегия не сгенерировала сигналов'}
            
            # Создаем индекс сигналов для быстрого поиска
            signals_by_time = {}
            for signal in signals:
                timestamp = signal.timestamp if hasattr(signal, 'timestamp') else signal['timestamp']
                if timestamp not in signals_by_time:
                    signals_by_time[timestamp] = []
                signals_by_time[timestamp].append(signal)
            
            # Основной цикл бэктестирования
            total_bars = len(data)
            for i, (timestamp, row) in enumerate(data.iterrows()):
                current_price = row['close']
                
                # Обработка сигналов для текущего времени
                if timestamp in signals_by_time:
                    for signal in signals_by_time[timestamp]:
                        if hasattr(signal, 'action'):
                            # TradeSignal объект
                            signal_dict = {
                                'type': signal.action,
                                'action': signal.action,
                                'price': signal.price,
                                'stop_loss': signal.stop_loss,
                                'take_profit': signal.take_profit,
                                'strategy': signal.strategy
                            }
                        else:
                            # Словарь
                            signal_dict = signal
                        
                        self.process_signal(signal_dict, current_price, timestamp)
                
                # Проверка условий выхода
                self.check_exit_conditions(current_price, timestamp)
                
                # Обновление капитала
                self.update_equity(current_price, timestamp)
                
                # Прогресс
                if verbose and i % (total_bars // 10) == 0:
                    progress = (i / total_bars) * 100
                    print(f"Прогресс бэктестирования: {progress:.0f}%")
            
            # Закрываем оставшиеся открытые позиции по последней цене
            final_price = data['close'].iloc[-1]
            final_timestamp = data.index[-1]
            
            for position in self.positions:
                if position.is_open:
                    trade = position.close_position(final_price, final_timestamp, "END_OF_DATA")
                    if trade:
                        costs = self.apply_costs(trade.quantity, final_price)
                        net_pnl = trade.pnl - costs
                        self.balance += net_pnl + (trade.quantity * trade.entry_price)
                        
                        if net_pnl > 0:
                            self.winning_trades += 1
                        else:
                            self.losing_trades += 1
                        
                        trade.pnl = net_pnl
                        if trade.entry_price != 0:
                            trade.pnl_pct = (net_pnl / (trade.quantity * trade.entry_price)) * 100
                        
                        self.completed_trades.append(trade)
            
            # Финальное обновление капитала
            self.update_equity(final_price, final_timestamp)
            
            # Расчет результатов
            return self.calculate_results(strategy_name)
            
        except Exception as e:
            return {'error': f'Ошибка бэктестирования: {str(e)}'}
    
    def calculate_results(self, strategy_name: str) -> Dict:
        """Расчет финальных результатов бэктестирования"""
        
        if not self.completed_trades:
            return {
                'error': 'Нет завершенных сделок',
                'total_trades': 0,
                'strategy_name': strategy_name
            }
        
        # Создаем DataFrame сделок
        trades_data = []
        for trade in self.completed_trades:
            trades_data.append({
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'direction': trade.direction,
                'pnl': trade.pnl,
                'pnl_pct': trade.pnl_pct,
                'duration_hours': trade.duration_hours,
                'exit_reason': trade.exit_reason,
                'strategy': trade.strategy
            })
        
        trades_df = pd.DataFrame(trades_data)
        
        # Создаем DataFrame кривой капитала
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('timestamp', inplace=True)
        
        # Основные метрики
        total_return = self.equity - self.initial_balance
        total_return_pct = (total_return / self.initial_balance) * 100
        
        # Метрики сделок
        total_trades = len(self.completed_trades)
        win_rate = self.winning_trades / total_trades if total_trades > 0 else 0
        
        # P&L статистика
        profits = trades_df[trades_df['pnl'] > 0]['pnl']
        losses = trades_df[trades_df['pnl'] <= 0]['pnl']
        
        avg_win = profits.mean() if len(profits) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        best_trade = trades_df['pnl'].max() if not trades_df.empty else 0
        worst_trade = trades_df['pnl'].min() if not trades_df.empty else 0
        
        # Профит-фактор
        total_profits = profits.sum() if len(profits) > 0 else 0
        total_losses = abs(losses.sum()) if len(losses) > 0 else 0
        profit_factor = total_profits / total_losses if total_losses > 0 else float('inf')
        
        # Максимальная просадка
        max_drawdown = equity_df['drawdown'].max() if not equity_df.empty else 0
        
        # Коэффициент Шарпа (упрощенный)
        if not trades_df.empty and trades_df['pnl_pct'].std() != 0:
            sharpe_ratio = trades_df['pnl_pct'].mean() / trades_df['pnl_pct'].std()
        else:
            sharpe_ratio = 0
        
        # Статистика по направлениям
        long_trades = trades_df[trades_df['direction'] == 'LONG']
        short_trades = trades_df[trades_df['direction'] == 'SHORT']
        
        long_winrate = len(long_trades[long_trades['pnl'] > 0]) / len(long_trades) if len(long_trades) > 0 else 0
        short_winrate = len(short_trades[short_trades['pnl'] > 0]) / len(short_trades) if len(short_trades) > 0 else 0
        
        # Причины выходов
        exit_reasons = trades_df['exit_reason'].value_counts().to_dict()
        
        # Серии побед/поражений
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        
        for trade in self.completed_trades:
            if trade.pnl > 0:
                consecutive_wins += 1
                consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            else:
                consecutive_losses += 1
                consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        
        return {
            'strategy_name': strategy_name,
            'initial_balance': self.initial_balance,
            'final_balance': self.equity,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'total_pnl': total_return,
            
            # Статистика сделок
            'total_trades': total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            
            # P&L метрики
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'profit_factor': profit_factor,
            
            # Риск метрики
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            
            # Временные характеристики
            'avg_trade_duration_hours': trades_df['duration_hours'].mean() if not trades_df.empty else 0,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            
            # Анализ по направлениям
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'long_winrate': long_winrate,
            'short_winrate': short_winrate,
            'long_pnl': long_trades['pnl'].sum() if not long_trades.empty else 0,
            'short_pnl': short_trades['pnl'].sum() if not short_trades.empty else 0,
            
            # Причины выходов
            'exit_reasons': exit_reasons,
            
            # DataFrames для дальнейшего анализа
            'trades_df': trades_df,
            'equity_curve': equity_df
        }
    
    def compare_strategies(self, data: pd.DataFrame, strategies: List, verbose: bool = True) -> Dict:
        """Сравнение нескольких стратегий"""
        results = {}
        
        for i, strategy in enumerate(strategies):
            if verbose:
                strategy_name = strategy.name if hasattr(strategy, 'name') else f"Strategy_{i+1}"
                print(f"\n📊 Тестирование стратегии: {strategy_name}")
            
            result = self.run_backtest(data, strategy, verbose=False)
            
            if 'error' not in result:
                strategy_name = result['strategy_name']
                results[strategy_name] = result
                
                if verbose:
                    print(f"  ✅ Доходность: {result['total_return_pct']:.2f}%")
                    print(f"  📊 Сделок: {result['total_trades']}")
                    print(f"  🎯 Винрейт: {result['win_rate']:.1%}")
            else:
                if verbose:
                    print(f"  ❌ Ошибка: {result['error']}")
        
        return results


# Пример использования
if __name__ == "__main__":
    print("🔄 Тестирование бэктестера...")
    
    # Создаем простую тестовую стратегию
    class SimpleStrategy:
        def __init__(self):
            self.name = "Simple_Test_Strategy"
        
        def generate_signals(self, df):
            from strategies import TradeSignal
            signals = []
            
            # Простая стратегия: покупаем на минимумах, продаем на максимумах
            for i in range(10, len(df), 20):  # Каждые 20 свечей
                if i < len(df):
                    timestamp = df.index[i]
                    price = df['close'].iloc[i]
                    
                    action = 'BUY' if i % 40 == 10 else 'SELL'
                    
                    signals.append(TradeSignal(
                        timestamp=timestamp,
                        action=action,
                        price=price,
                        confidence=0.7,
                        strategy=self.name,
                        reason=f"Тестовый сигнал {action}"
                    ))
            
            return signals
    
    # Создаем тестовые данные
    import pandas as pd
    from datetime import datetime, timedelta
    
    dates = pd.date_range(start='2023-01-01', end='2023-06-01', freq='1H')
    np.random.seed(42)
    
    # Генерируем случайные цены
    prices = [1000]
    for _ in range(len(dates) - 1):
        change = np.random.normal(0, 0.02)
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 500))  # Минимальная цена
    
    test_data = pd.DataFrame({
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': np.random.randint(100, 1000, len(dates))
    }, index=dates)
    
    # Тестируем бэктестер
    backtester = Backtester(initial_balance=10000, commission=0.001)
    strategy = SimpleStrategy()
    
    print("📊 Запуск тестового бэктестирования...")
    result = backtester.run_backtest(test_data, strategy)
    
    if 'error' not in result:
        print(f"\n✅ Результаты тестирования:")
        print(f"💰 Начальный капитал: ${result['initial_balance']:,.2f}")
        print(f"💰 Конечный капитал: ${result['final_balance']:,.2f}")
        print(f"📈 Доходность: {result['total_return_pct']:.2f}%")
        print(f"📊 Всего сделок: {result['total_trades']}")
        print(f"🎯 Винрейт: {result['win_rate']:.1%}")
        print(f"📉 Макс. просадка: {result['max_drawdown']:.2f}%")
    else:
        print(f"❌ Ошибка: {result['error']}")
    
    print("\n✅ Тестирование бэктестера завершено!")