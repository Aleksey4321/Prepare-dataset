#!/usr/bin/env python3

import os
import sys
import json
import logging
from typing import List

# Автоматическая установка всех необходимых зависимостей
REQUIRED_PACKAGES = [
    "tqdm",
    "aiohttp",
    "tenacity",
    "g4f"
]

def install_and_import(package):
    import importlib
    try:
        importlib.import_module(package)
    except ImportError:
        logging.info(f"Устанавливаю пакет: {package}")
        os.system(f"{sys.executable} -m pip install {package}")
        try:
            importlib.import_module(package)
        except ImportError:
            logging.error(f"Не удалось установить пакет: {package}")
            sys.exit(1)

for pkg in REQUIRED_PACKAGES:
    install_and_import(pkg)

import tqdm
import asyncio
import aiohttp
import g4f
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import Dict, Any, Tuple
import re
import time
import threading

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler("parse_and_prepare_dataset.log"), logging.StreamHandler()]
)

INPUT_FILE = "daily_dialog_dask_train.txt"
INTERMEDIATE_FILE = "dialogs_for_translate.txt"
ROUTER_PHRASE_OUTPUT_FILE = "router_phrase_dataset.json"
ROUTER_DIALOG_OUTPUT_FILE = "router_dialog_dataset.json"
MAIN_OUTPUT_FILE = "main_dataset.json"
MAX_CONCURRENT_REQUESTS = 5
RETRY_ATTEMPTS = 3

# Глобальный список тем для консистентности
KNOWN_TOPICS = set()

def parse_dialogs(input_path: str, output_path: str):
    """Парсит все диалоги из файла без ограничений"""
    dialogs = []
    total_lines = 0
    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            total_lines = i + 1
            try:
                obj = json.loads(line)
                prompt = obj.get("prompt", [])
                dialogs.append(prompt)
            except Exception as e:
                logging.error(f"Ошибка парсинга строки {i+1}: {e}")
            
            # Логируем прогресс каждые 10000 строк
            if (i + 1) % 10000 == 0:
                logging.info(f"Парсинг: обработано строк {i+1}, найдено диалогов: {len(dialogs)}")
    
    logging.info(f"Общий парсинг завершён: {total_lines} строк, {len(dialogs)} диалогов")

    # Сохраняем в промежуточный текстовый файл
    with open(output_path, "w", encoding="utf-8") as out:
        for idx, dialog in enumerate(dialogs, 1):
            out.write("——————\n\nДиалог:\n\n")
            for replica in dialog:
                role = replica.get("role", "unknown")
                text = replica.get("content", {}).get("dialog", "")
                out.write(f"{role}: {text}\n")
            out.write("\n——————\n\n")
            
            # Логируем прогресс сохранения каждые 5000 диалогов
            if idx % 5000 == 0:
                logging.info(f"Сохранено диалогов в промежуточный файл: {idx}")
    
    logging.info(f"Промежуточный файл сохранён: {output_path}")
    return dialogs


def create_translation_prompt(dialog_text: str, known_topics: list) -> str:
    """Создаёт промт для GPT-4 для перевода и определения темы диалога"""
    topics_str = ", ".join(known_topics) if known_topics else "Нет предыдущих тем"
    
    prompt = f"""Проанализируй следующий диалог на английском языке и выполни ТРИ задачи:

1. ПЕРЕВЕДИ диалог на русский язык, сохранив естественность речи и смысл
2. ОПРЕДЕЛИ одну тему диалога средней специфичности (не слишком узкую, не слишком широкую)
3. НАЙДИ одну конкретную реплику пользователя (user), которая максимально прямо, явно, очевидно и ясно отражает основную центральную главную тему диалога, а также переведи её на русский язык. 

ДИАЛОГ ДЛЯ АНАЛИЗА:
{dialog_text}

РАНЕЕ ОПРЕДЕЛЁННЫЕ ТЕМЫ: {topics_str}

ВАЖНО: 
- Если диалог подходит под одну из ранее определённых тем, используй её
- Тема должна быть средней специфичности. Она не должна быть слишком узкой, но и не должна быть слишком широкой. 
- НЕ используй подчеркивания в названии темы
- Тема должна быть 1-3 слова
- Ключевая фраза должна ПРЯМО отражать тему. Если такой фразы нет, фразы отражают тему лишь косвенно, не прямо, не очевидно, не явно, нет одной конкретной реплики которая на 100% отражает ключевую тему диалога - верни "none". ОБЯЗАТЕЛЬНО ПЕРЕВЕДИ ФРАЗУ НА РУССКИЙ ЯЗЫК. КРАЙНЕ ВАЖНО ПЕРЕВЕСТИ КЛЮЧЕВУЮ ФРАЗУ НА РУССКИЙ ЯЗЫК И НАЗВАТЬ ЕЁ НА РУССКОМ ЯЗЫКЕ, А НЕ НА АНГЛИЙСКОМ. 

ВЕРНИ ОТВЕТ СТРОГО В СЛЕДУЮЩЕМ ФОРМАТЕ (без дополнительных пояснений):

ТЕМА: [название темы без подчеркиваний]
КЛЮЧЕВАЯ_ФРАЗА: [одна фраза от пользователя, которая прямо отражает тему, или "none". Тема должна быть переведена на русский язык]
ПЕРЕВЕДЁННЫЙ_ДИАЛОГ:
[переведённый диалог с сохранением ролей]

Пример формата ответа:
ТЕМА: повседневное общение
КЛЮЧЕВАЯ_ФРАЗА: Привет, как дела?
ПЕРЕВЕДЁННЫЙ_ДИАЛОГ:
user: Привет, как дела?
assistant: Привет! Всё отлично, а у тебя как?
user: Нормально, спасибо за вопрос."""

    return prompt


@retry(stop=stop_after_attempt(RETRY_ATTEMPTS), wait=wait_exponential(multiplier=1, min=4, max=10))
async def translate_and_categorize_dialog(session, dialog_text: str, known_topics: list) -> Tuple[str, str, str]:
    """Асинхронно переводит диалог и определяет его тему через GPT-4"""
    try:
        prompt = create_translation_prompt(dialog_text, known_topics)
        
        # Используем g4f для запроса к GPT-4
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: g4f.ChatCompletion.create(
                model=g4f.models.gemini_2_5_flash,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
            )
        )
        
        if not response:
            raise Exception("Пустой ответ от GPT-4")
            
        # Парсим ответ
        topic, key_phrase, translated_dialog = parse_gpt_response(response)
        
        logging.info(f"Обработан диалог, тема: {topic}, ключевая фраза: {key_phrase}")
        return topic, key_phrase, translated_dialog, dialog_text
        
    except Exception as e:
        logging.error(f"Ошибка при переводе диалога: {e}")
        raise


def parse_gpt_response(response: str) -> Tuple[str, str, str]:
    """Парсит ответ GPT-4 и извлекает тему, ключевую фразу и переведённый диалог"""
    try:
        lines = response.strip().split('\n')
        topic = ""
        key_phrase = ""
        translated_dialog = ""
        dialog_started = False
        
        for line in lines:
            if line.startswith("ТЕМА:"):
                topic = line.replace("ТЕМА:", "").strip()
            elif line.startswith("КЛЮЧЕВАЯ_ФРАЗА:"):
                key_phrase = line.replace("КЛЮЧЕВАЯ_ФРАЗА:", "").strip()
            elif line.startswith("ПЕРЕВЕДЁННЫЙ_ДИАЛОГ:"):
                dialog_started = True
            elif dialog_started and line.strip():
                translated_dialog += line + "\n"
        
        if not topic:
            # Попытка найти тему в другом формате
            topic_match = re.search(r'ТЕМА[:\s]+([^\n]+)', response)
            if topic_match:
                topic = topic_match.group(1).strip()
            else:
                topic = "общение"  # дефолтная тема
        
        if not key_phrase:
            # Попытка найти ключевую фразу в другом формате
            key_phrase_match = re.search(r'КЛЮЧЕВАЯ_ФРАЗА[:\s]+([^\n]+)', response)
            if key_phrase_match:
                key_phrase = key_phrase_match.group(1).strip()
            else:
                key_phrase = "none"
        
        if not translated_dialog.strip():
            # Если диалог не найден, берём всё после "ПЕРЕВЕДЁННЫЙ_ДИАЛОГ:"
            dialog_match = re.search(r'ПЕРЕВЕДЁННЫЙ_ДИАЛОГ[:\s]*\n(.*)', response, re.DOTALL)
            if dialog_match:
                translated_dialog = dialog_match.group(1).strip()
            else:
                translated_dialog = "Ошибка перевода"
        
        return topic.lower().replace("_", " "), key_phrase, translated_dialog.strip()
        
    except Exception as e:
        logging.error(f"Ошибка парсинга ответа GPT: {e}")
        return "общение", "none", response  # возвращаем исходный ответ как fallback


async def process_dialogs_async(dialogs: List[List[Dict]], max_concurrent: int = MAX_CONCURRENT_REQUESTS):
    """Асинхронно обрабатывает все диалоги"""
    logging.info(f"Начинаю асинхронную обработку {len(dialogs)} диалогов")
    
    # Семафор для ограничения количества одновременных запросов
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_single_dialog(session, dialog, index):
        async with semaphore:
            try:
                # Формируем текст диалога
                dialog_text = ""
                for replica in dialog:
                    role = replica.get("role", "unknown")
                    text = replica.get("content", {}).get("dialog", "")
                    dialog_text += f"{role}: {text}\n"
                
                # Обрабатываем диалог
                topic, key_phrase, translated_dialog, original_dialog = await translate_and_categorize_dialog(
                    session, dialog_text, list(KNOWN_TOPICS)
                )
                
                # Добавляем тему в глобальный список
                KNOWN_TOPICS.add(topic)
                
                return {
                    'index': index,
                    'topic': topic,
                    'key_phrase': key_phrase,
                    'translated_dialog': translated_dialog,
                    'original_dialog': original_dialog,
                    'success': True
                }
                
            except Exception as e:
                logging.error(f"Ошибка обработки диалога {index}: {e}")
                return {
                    'index': index,
                    'success': False,
                    'error': str(e)
                }
    
    # Создаём сессию для HTTP-запросов (если понадобится)
    async with aiohttp.ClientSession() as session:
        # Создаём ВСЕ задачи одновременно
        tasks = [
            process_single_dialog(session, dialog, i) 
            for i, dialog in enumerate(dialogs)
        ]
        
        logging.info(f"Запускаю {len(tasks)} задач одновременно")
        
        # Запускаем ВСЕ задачи одновременно с gather()
        with tqdm.tqdm(total=len(tasks), desc="Обработка диалогов") as pbar:
            # Ждём завершения ВСЕХ задач одновременно
            results = await asyncio.gather(*tasks, return_exceptions=True)
            pbar.update(len(tasks))  # Обновляем прогресс-бар сразу после завершения
        
        # Обрабатываем результаты
        successful_results = []
        failed_count = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logging.error(f"Задача {i} завершилась с исключением: {result}")
                failed_count += 1
            elif result.get('success', False):
                successful_results.append(result)
            else:
                failed_count += 1
        
        # Сортируем результаты по индексу
        successful_results.sort(key=lambda x: x['index'])
        
        logging.info(f"Обработано успешно: {len(successful_results)}, неудачно: {failed_count}")
        
        return successful_results


def create_router_phrase_dataset(processed_dialogs: List[Dict]) -> List[Dict]:
    """Создаёт датасет для роутера с ключевыми фразами"""
    router_data = []
    
    for dialog_data in processed_dialogs:
        try:
            topic = dialog_data['topic']
            key_phrase = dialog_data['key_phrase']
            
            # Добавляем только если есть ключевая фраза
            if key_phrase and key_phrase.lower() != "none":
                router_entry = {
                    "network_type": "router",
                    "input": key_phrase,
                    "topics": [topic],
                    "language": "russian"
                }
                router_data.append(router_entry)
                
        except Exception as e:
            logging.error(f"Ошибка создания записи роутера (фраза): {e}")
    
    logging.info(f"Создано записей для роутера (фразы): {len(router_data)}")
    return router_data


def create_router_dialog_dataset(processed_dialogs: List[Dict]) -> List[Dict]:
    """Создаёт датасет для роутера с полными диалогами"""
    router_data = []
    
    for dialog_data in processed_dialogs:
        try:
            topic = dialog_data['topic']
            translated_dialog = dialog_data['translated_dialog']
            
            # Форматируем полный диалог
            lines = translated_dialog.strip().split('\n')
            formatted_dialog = ""
            
            for line in lines:
                if ':' in line:
                    role_part, content_part = line.split(':', 1)
                    role = role_part.strip()
                    content = content_part.strip()
                    
                    if role == 'user':
                        formatted_dialog += f"User: {content}\n"
                    elif role == 'assistant':
                        formatted_dialog += f"Assistant: {content}\n"
            
            if formatted_dialog.strip():
                router_entry = {
                    "network_type": "router",
                    "input": formatted_dialog.strip(),
                    "topics": [topic],
                    "language": "russian"
                }
                router_data.append(router_entry)
                
        except Exception as e:
            logging.error(f"Ошибка создания записи роутера (диалог): {e}")
    
    logging.info(f"Создано записей для роутера (диалоги): {len(router_data)}")
    return router_data


def create_main_dataset(processed_dialogs: List[Dict]) -> List[Dict]:
    """Создаёт датасет для главной сети"""
    main_data = []
    
    for dialog_data in processed_dialogs:
        try:
            topic = dialog_data['topic']
            translated_dialog = dialog_data['translated_dialog']
            
            # Парсим диалог и создаём запись типа "dialog"
            messages = []
            lines = translated_dialog.strip().split('\n')
            
            for line in lines:
                if ':' in line:
                    role_part, content_part = line.split(':', 1)
                    role = role_part.strip()
                    content = content_part.strip()
                    
                    if role in ['user', 'assistant'] and content:
                        message = {
                            "role": role,
                            "content": content
                        }
                        messages.append(message)
            
            if messages:
                dialog_entry = {
                    "network_type": "main",
                    "type": "dialog",
                    "topic": topic,
                    "messages": messages
                }
                main_data.append(dialog_entry)
                
        except Exception as e:
            logging.error(f"Ошибка создания записи главной сети: {e}")
    
    logging.info(f"Создано записей для главной сети: {len(main_data)}")
    return main_data


def save_json_dataset(data: List[Dict], filename: str):
    """Сохраняет датасет в JSON файл"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logging.info(f"Датасет сохранён: {filename} ({len(data)} записей)")
    except Exception as e:
        logging.error(f"Ошибка сохранения {filename}: {e}")


async def main_processing():
    """Основная функция обработки"""
    # 1. Парсинг исходного файла
    if not os.path.exists(INPUT_FILE):
        logging.error(f"Файл {INPUT_FILE} не найден!")
        return False
    
    logging.info("Этап 1: Парсинг исходного файла")
    dialogs = []
    total_lines = 0
    
    # Для тестирования ограничим количество диалогов
    TEST_LIMIT = getattr(sys.modules[__name__], 'MAX_TEST_DIALOGS', None) or 1000000
    
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if len(dialogs) >= TEST_LIMIT:
                break
                
            total_lines = i + 1
            try:
                obj = json.loads(line)
                prompt = obj.get("prompt", [])
                if prompt:  # только непустые диалоги
                    dialogs.append(prompt)
            except Exception as e:
                logging.error(f"Ошибка парсинга строки {i+1}: {e}")
            
            # Логируем прогресс каждые 10000 строк
            if (i + 1) % 10000 == 0:
                logging.info(f"Обработано строк: {i+1}, найдено диалогов: {len(dialogs)}")
    
    logging.info(f"Всего обработано строк: {total_lines}")
    logging.info(f"Извлечено диалогов: {len(dialogs)} (лимит для теста: {TEST_LIMIT})")
    
    if not dialogs:
        logging.error("Не найдено диалогов для обработки!")
        return False
    
    # 2. Асинхронная обработка диалогов
    logging.info("Этап 2: Перевод и определение тем")
    processed_dialogs = await process_dialogs_async(dialogs)
    
    if not processed_dialogs:
        logging.error("Не удалось обработать ни одного диалога!")
        return False
    
    # 3. Создание датасетов
    logging.info("Этап 3: Формирование итоговых датасетов")
    router_phrase_data = create_router_phrase_dataset(processed_dialogs)
    router_dialog_data = create_router_dialog_dataset(processed_dialogs)
    main_data = create_main_dataset(processed_dialogs)
    
    # 4. Сохранение файлов
    logging.info("Этап 4: Сохранение файлов")
    save_json_dataset(router_phrase_data, ROUTER_PHRASE_OUTPUT_FILE)
    save_json_dataset(router_dialog_data, ROUTER_DIALOG_OUTPUT_FILE)
    save_json_dataset(main_data, MAIN_OUTPUT_FILE)
    
    logging.info(f"Обработка завершена! Найдено тем: {len(KNOWN_TOPICS)}")
    logging.info(f"Темы: {', '.join(sorted(KNOWN_TOPICS))}")
    
    return True

if __name__ == "__main__":
    try:
        # Запускаем основную асинхронную обработку
        success = asyncio.run(main_processing())
        if success:
            logging.info("Скрипт выполнен успешно!")
        else:
            logging.error("Скрипт завершился с ошибками!")
            sys.exit(1)
    except KeyboardInterrupt:
        logging.info("Обработка прервана пользователем")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Критическая ошибка: {e}")
        sys.exit(1)