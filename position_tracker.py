import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import json
import os

@dataclass
class RealPosition:
    """Класс для реальной торговой позиции"""
    id: str
    symbol: str
    direction: str  # 'LONG' или 'SHORT'
    entry_price: float
    quantity: float
    entry_time: datetime
    strategy: str
    confidence: float
    
    # Уровни риск-менеджмента
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None
    
    # Текущее состояние
    current_price: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    unrealized_pnl_pct: Optional[float] = None
    
    # Статус позиции
    is_active: bool = True
    exit_alerts: List[str] = None
    last_update: Optional[datetime] = None
    
    def __post_init__(self):
        if self.exit_alerts is None:
            self.exit_alerts = []

class PositionTracker:
    """Система отслеживания реальных торговых позиций"""
    
    def __init__(self, data_file: str = "positions.json"):
        self.data_file = data_file
        self.positions: Dict[str, RealPosition] = {}
        self.load_positions()
        
        # Параметры риск-менеджмента
        self.default_stop_loss_pct = 3.0  # 3% стоп-лосс
        self.default_take_profit_pct = 6.0  # 6% тейк-профит (1:2 соотношение)
        self.trailing_stop_activation = 4.0  # Активация трейлинг-стопа при 4% прибыли
        self.trailing_stop_distance = 2.0  # Дистанция трейлинг-стопа 2%
    
    def generate_position_id(self) -> str:
        """Генерация уникального ID позиции"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"POS_{timestamp}_{len(self.positions):03d}"
    
    def add_position(self, symbol: str, direction: str, entry_price: float, 
                    quantity: float, strategy: str, confidence: float,
                    custom_stop_loss: Optional[float] = None,
                    custom_take_profit: Optional[float] = None) -> str:
        """Добавление новой позиции"""
        
        position_id = self.generate_position_id()
        
        # Расчет уровней стоп-лосса и тейк-профита
        if direction == 'LONG':
            if custom_stop_loss is None:
                stop_loss = entry_price * (1 - self.default_stop_loss_pct / 100)
            else:
                stop_loss = custom_stop_loss
                
            if custom_take_profit is None:
                take_profit = entry_price * (1 + self.default_take_profit_pct / 100)
            else:
                take_profit = custom_take_profit
        else:  # SHORT
            if custom_stop_loss is None:
                stop_loss = entry_price * (1 + self.default_stop_loss_pct / 100)
            else:
                stop_loss = custom_stop_loss
                
            if custom_take_profit is None:
                take_profit = entry_price * (1 - self.default_take_profit_pct / 100)
            else:
                take_profit = custom_take_profit
        
        position = RealPosition(
            id=position_id,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            quantity=quantity,
            entry_time=datetime.now(),
            strategy=strategy,
            confidence=confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            current_price=entry_price,
            last_update=datetime.now()
        )
        
        self.positions[position_id] = position
        self.save_positions()
        
        return position_id
    
    def update_position_prices(self, current_prices: Dict[str, float]):
        """Обновление текущих цен для всех активных позиций"""
        
        for position_id, position in self.positions.items():
            if not position.is_active:
                continue
                
            symbol_clean = position.symbol.replace('/', '')
            if symbol_clean in current_prices:
                position.current_price = current_prices[symbol_clean]
                position.last_update = datetime.now()
                
                # Расчет нереализованной прибыли/убытка
                if position.direction == 'LONG':
                    position.unrealized_pnl = (position.current_price - position.entry_price) * position.quantity
                    position.unrealized_pnl_pct = ((position.current_price / position.entry_price) - 1) * 100
                else:  # SHORT
                    position.unrealized_pnl = (position.entry_price - position.current_price) * position.quantity
                    position.unrealized_pnl_pct = ((position.entry_price / position.current_price) - 1) * 100
                
                # Проверка условий выхода
                self._check_exit_conditions(position)
                
                # Обновление трейлинг-стопа
                self._update_trailing_stop(position)
        
        self.save_positions()
    
    def _check_exit_conditions(self, position: RealPosition):
        """Проверка условий для выхода из позиции"""
        
        if not position.current_price:
            return
        
        alerts = []
        
        # Проверка стоп-лосса
        if position.stop_loss:
            if position.direction == 'LONG' and position.current_price <= position.stop_loss:
                alerts.append(f"🛑 СТОП-ЛОСС! Цена {position.current_price:.4f} достигла уровня {position.stop_loss:.4f}")
            elif position.direction == 'SHORT' and position.current_price >= position.stop_loss:
                alerts.append(f"🛑 СТОП-ЛОСС! Цена {position.current_price:.4f} достигла уровня {position.stop_loss:.4f}")
        
        # Проверка тейк-профита
        if position.take_profit:
            if position.direction == 'LONG' and position.current_price >= position.take_profit:
                alerts.append(f"🎯 ТЕЙК-ПРОФИТ! Цена {position.current_price:.4f} достигла цели {position.take_profit:.4f}")
            elif position.direction == 'SHORT' and position.current_price <= position.take_profit:
                alerts.append(f"🎯 ТЕЙК-ПРОФИТ! Цена {position.current_price:.4f} достигла цели {position.take_profit:.4f}")
        
        # Проверка трейлинг-стопа
        if position.trailing_stop:
            if position.direction == 'LONG' and position.current_price <= position.trailing_stop:
                alerts.append(f"📉 ТРЕЙЛИНГ-СТОП! Цена {position.current_price:.4f} упала до {position.trailing_stop:.4f}")
            elif position.direction == 'SHORT' and position.current_price >= position.trailing_stop:
                alerts.append(f"📈 ТРЕЙЛИНГ-СТОП! Цена {position.current_price:.4f} выросла до {position.trailing_stop:.4f}")
        
        # Дополнительные предупреждения
        if position.unrealized_pnl_pct:
            # Предупреждение о крупных убытках
            if position.unrealized_pnl_pct < -5:
                alerts.append(f"⚠️ КРУПНЫЙ УБЫТОК: {position.unrealized_pnl_pct:.2f}% - рассмотрите выход")
            
            # Уведомление о хорошей прибыли
            elif position.unrealized_pnl_pct > 8:
                alerts.append(f"💰 ОТЛИЧНАЯ ПРИБЫЛЬ: {position.unrealized_pnl_pct:.2f}% - рассмотрите частичную фиксацию")
        
        # Добавляем новые алерты (избегаем дублирования)
        for alert in alerts:
            if alert not in position.exit_alerts:
                position.exit_alerts.append(alert)
    
    def _update_trailing_stop(self, position: RealPosition):
        """Обновление трейлинг-стопа"""
        
        if not position.current_price or not position.unrealized_pnl_pct:
            return
        
        # Активируем трейлинг-стоп только при достижении определенной прибыли
        if position.unrealized_pnl_pct >= self.trailing_stop_activation:
            
            if position.direction == 'LONG':
                new_trailing_stop = position.current_price * (1 - self.trailing_stop_distance / 100)
                # Обновляем только если новый уровень выше текущего
                if not position.trailing_stop or new_trailing_stop > position.trailing_stop:
                    position.trailing_stop = new_trailing_stop
            
            else:  # SHORT
                new_trailing_stop = position.current_price * (1 + self.trailing_stop_distance / 100)
                # Обновляем только если новый уровень ниже текущего
                if not position.trailing_stop or new_trailing_stop < position.trailing_stop:
                    position.trailing_stop = new_trailing_stop
    
    def close_position(self, position_id: str, exit_price: float, reason: str = "Manual"):
        """Закрытие позиции"""
        
        if position_id in self.positions:
            position = self.positions[position_id]
            position.is_active = False
            position.current_price = exit_price
            
            # Финальный расчет P&L
            if position.direction == 'LONG':
                final_pnl = (exit_price - position.entry_price) * position.quantity
                final_pnl_pct = ((exit_price / position.entry_price) - 1) * 100
            else:  # SHORT
                final_pnl = (position.entry_price - exit_price) * position.quantity
                final_pnl_pct = ((position.entry_price / exit_price) - 1) * 100
            
            position.unrealized_pnl = final_pnl
            position.unrealized_pnl_pct = final_pnl_pct
            position.exit_alerts.append(f"✅ ПОЗИЦИЯ ЗАКРЫТА: {reason} | P&L: {final_pnl_pct:.2f}%")
            
            self.save_positions()
            return True
        
        return False
    
    def get_active_positions(self) -> List[RealPosition]:
        """Получение всех активных позиций"""
        return [pos for pos in self.positions.values() if pos.is_active]
    
    def get_position_summary(self) -> Dict:
        """Сводка по всем позициям"""
        
        try:
            active_positions = self.get_active_positions()
            closed_positions = [pos for pos in self.positions.values() if not pos.is_active]
            
            # Базовая структура с обязательными ключами
            summary = {
                'total_positions': len(self.positions),
                'active_positions': len(active_positions),
                'closed_positions': len(closed_positions),
                'total_unrealized_pnl': 0.0,
                'total_unrealized_pnl_pct': 0.0,
                'alerts_count': 0,
                'win_rate': 0.0
            }
            
            if not active_positions and not closed_positions:
                return summary
        
            # Активные позиции
            if active_positions:
                total_unrealized_pnl = sum(pos.unrealized_pnl or 0 for pos in active_positions)
                total_entry_value = sum(pos.entry_price * pos.quantity for pos in active_positions if pos.entry_price and pos.quantity)
                
                if total_entry_value > 0:
                    summary['total_unrealized_pnl'] = total_unrealized_pnl
                    summary['total_unrealized_pnl_pct'] = (total_unrealized_pnl / total_entry_value) * 100
                
                # Подсчет алертов
                summary['alerts_count'] = sum(len(pos.exit_alerts or []) for pos in active_positions)
            
            # Статистика по закрытым позициям
            if closed_positions:
                closed_pnl = [pos.unrealized_pnl_pct for pos in closed_positions if pos.unrealized_pnl_pct is not None]
                if closed_pnl:
                    winning_positions = len([pnl for pnl in closed_pnl if pnl > 0])
                    summary['win_rate'] = (winning_positions / len(closed_pnl)) * 100
            
            return summary
            
        except Exception as e:
            print(f"Ошибка в get_position_summary: {e}")
            # Возвращаем безопасную структуру
            return {
                'total_positions': 0,
                'active_positions': 0,
                'closed_positions': 0,
                'total_unrealized_pnl': 0.0,
                'total_unrealized_pnl_pct': 0.0,
                'alerts_count': 0,
                'win_rate': 0.0
            }
    
    def get_risk_recommendations(self, symbol: str, entry_price: float, 
                               direction: str, account_balance: float = 10000) -> Dict:
        """Получение рекомендаций по риск-менеджменту"""
        
        risk_per_trade = 0.02  # 2% риска на сделку
        
        if direction == 'LONG':
            recommended_stop_loss = entry_price * (1 - self.default_stop_loss_pct / 100)
            recommended_take_profit = entry_price * (1 + self.default_take_profit_pct / 100)
        else:  # SHORT
            recommended_stop_loss = entry_price * (1 + self.default_stop_loss_pct / 100)
            recommended_take_profit = entry_price * (1 - self.default_take_profit_pct / 100)
        
        # Расчет размера позиции
        risk_amount = account_balance * risk_per_trade
        price_risk = abs(entry_price - recommended_stop_loss)
        position_size = risk_amount / price_risk if price_risk > 0 else 0
        
        return {
            'recommended_stop_loss': recommended_stop_loss,
            'recommended_take_profit': recommended_take_profit,
            'position_size': position_size,
            'risk_amount': risk_amount,
            'risk_reward_ratio': abs(recommended_take_profit - entry_price) / abs(entry_price - recommended_stop_loss) if price_risk > 0 else 0,
            'notes': [
                f"💡 Рекомендуемый размер позиции: {position_size:.4f} {symbol.split('/')[0]}",
                f"🛡️ Стоп-лосс: ${recommended_stop_loss:.4f} ({self.default_stop_loss_pct}%)",
                f"🎯 Тейк-профит: ${recommended_take_profit:.4f} ({self.default_take_profit_pct}%)",
                f"📊 Риск на сделку: ${risk_amount:.2f} ({risk_per_trade*100}% от баланса)",
                f"⚖️ Соотношение риск/прибыль: 1:{abs(recommended_take_profit - entry_price) / abs(entry_price - recommended_stop_loss):.1f}" if price_risk > 0 else ""
            ]
        }
    
    def get_real_trade_history(self) -> List[Dict]:
        """Получение истории только реальных сделок (без автоматических сигналов)"""
        
        real_trades = []
        for position in self.positions.values():
            if not position.is_active:  # Только закрытые позиции
                real_trades.append({
                    'timestamp': position.entry_time,
                    'symbol': position.symbol,
                    'direction': position.direction,
                    'entry_price': position.entry_price,
                    'exit_price': position.current_price,
                    'pnl_pct': position.unrealized_pnl_pct,
                    'strategy': position.strategy,
                    'duration': (datetime.now() - position.entry_time).days if position.is_active else 
                               (position.last_update - position.entry_time).days if position.last_update else 0
                })
        
        # Сортируем по времени (новые сначала)
        real_trades.sort(key=lambda x: x['timestamp'], reverse=True)
        return real_trades
    
    def save_positions(self):
        """Сохранение позиций в файл"""
        try:
            # Конвертируем datetime в строки для JSON
            positions_data = {}
            for pos_id, position in self.positions.items():
                pos_dict = asdict(position)
                pos_dict['entry_time'] = position.entry_time.isoformat()
                if position.last_update:
                    pos_dict['last_update'] = position.last_update.isoformat()
                positions_data[pos_id] = pos_dict
            
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(positions_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Ошибка сохранения позиций: {e}")
    
    def load_positions(self):
        """Загрузка позиций из файла"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    positions_data = json.load(f)
                
                for pos_id, pos_dict in positions_data.items():
                    try:
                        # Конвертируем строки обратно в datetime
                        pos_dict['entry_time'] = datetime.fromisoformat(pos_dict['entry_time'])
                        if pos_dict.get('last_update'):
                            pos_dict['last_update'] = datetime.fromisoformat(pos_dict['last_update'])
                        
                        # Проверяем обязательные поля
                        required_fields = ['id', 'symbol', 'direction', 'entry_price', 'quantity', 'strategy', 'confidence']
                        if all(field in pos_dict for field in required_fields):
                            self.positions[pos_id] = RealPosition(**pos_dict)
                        else:
                            print(f"Пропущена позиция {pos_id}: отсутствуют обязательные поля")
                            
                    except Exception as e:
                        print(f"Ошибка загрузки позиции {pos_id}: {e}")
                        continue
                        
        except Exception as e:
            print(f"Ошибка загрузки позиций: {e}")
            self.positions = {}


# Пример использования
if __name__ == "__main__":
    print("🔄 Тестирование трекера позиций...")
    
    # Создаем трекер
    tracker = PositionTracker()
    
    # Добавляем тестовую позицию
    position_id = tracker.add_position(
        symbol="BTC/USDT",
        direction="LONG",
        entry_price=50000,
        quantity=0.1,
        strategy="trend_following",
        confidence=85
    )
    
    print(f"✅ Добавлена позиция: {position_id}")
    print("✅ Файл position_tracker.py готов!")