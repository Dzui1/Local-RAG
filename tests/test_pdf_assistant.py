import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from courses_rag.assistant import Assistant, batches
from courses_rag.cli import chat, main
from courses_rag.library import Library


class FakeEmbedder:
    def encode(self, texts):
        if isinstance(texts, str):
            return np.array([1.0, 0.0, 0.0])
        return np.array([[1.0, 0.0, 0.0] for _ in texts])


class LibraryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / 'data'
        self.data.mkdir()
        self.library = Library(self.data, self.root / 'db', FakeEmbedder())

    def tearDown(self):
        self.temp.cleanup()

    def pdf(self, name, content=b'first version'):
        path = self.data / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def index(self, topic, text):
        reader = Mock(pages=[Mock(extract_text=Mock(return_value=text))])
        with patch('pypdf.PdfReader', return_value=reader), patch('sys.stdout', new=io.StringIO()):
            return self.library.index(topic)

    def test_discovery_and_topic_isolation_with_same_filenames(self):
        self.pdf('cities/same.PDF')
        self.pdf('biology/same.PDF')
        self.pdf('root.pdf')
        self.assertEqual(set(self.library.topics()), {'.', 'cities', 'biology'})
        self.index('cities', 'urban planning')
        self.index('biology', 'cell division')
        self.assertEqual(self.library.search('cities', 'meaning')[0][0], 'urban planning')
        self.assertEqual(self.library.search('biology', 'meaning')[0][0], 'cell division')
        with self.assertRaises(ValueError):
            self.library.files('../outside')

    def test_incremental_updates_deletions_and_pdf_selection(self):
        first = self.pdf('one.pdf')
        self.index('.', 'original text')
        with patch('pypdf.PdfReader', side_effect=AssertionError('unchanged PDF read')):
            self.library.index('.')
        first.write_bytes(b'changed')
        self.index('.', 'replacement text')
        self.pdf('two.pdf')
        self.index('.', 'second document')
        self.assertEqual(self.library.search('.', 'meaning', 'one.pdf')[0][0], 'replacement text')
        first.unlink()
        self.index('.', 'unused')
        self.assertEqual({m['source'] for _, m in self.library.records('.')}, {'two.pdf'})
        with self.assertRaises(ValueError):
            self.library.records('.', 'one.pdf')

    def test_scanned_pdf_has_clear_error(self):
        self.pdf('scan.pdf')
        self.index('.', '')
        with self.assertRaisesRegex(ValueError, 'OCR'):
            self.library.records('.')

    def test_incomplete_index_is_rebuilt(self):
        self.pdf('one.pdf')
        collection = self.index('.', 'word ' * 500)
        total = collection.count()
        collection.delete(ids=collection.get()['ids'][:1])
        self.assertEqual(collection.count(), total - 1)
        self.index('.', 'word ' * 500)
        self.assertEqual(collection.count(), total)

    def test_default_chat_uses_available_folder_without_root_pdfs(self):
        self.pdf('biology/one.pdf')
        with patch('courses_rag.cli.chat') as run, patch('sys.stdout', new=io.StringIO()):
            main(['chat', '--data', str(self.data), '--db', str(self.root / 'db')])
        self.assertEqual(run.call_args.args[2], 'biology')


class AssistantTests(unittest.TestCase):
    def test_followups_and_topic_histories_are_isolated(self):
        library = Mock()
        library.search.return_value = [('evidence', {'source': 'a.pdf', 'page': 1})]
        assistant = Assistant(library)
        assistant.generate = Mock(return_value='first answer')
        answer = assistant.ask('cities', 'What is urban theory?')
        self.assertIn('a.pdf — PDF pages 1', answer)
        assistant.ask('biology', 'What is a cell?')
        self.assertEqual(assistant.generate.call_args.args[1], [])
        assistant.ask('cities', 'What does that mean?')
        self.assertIn('urban theory', library.search.call_args.args[1])
        self.assertNotIn('cell', library.search.call_args.args[1])
        self.assertEqual(assistant.generate.call_args.args[1][0]['content'], 'What is urban theory?')

    def test_summary_reads_every_chunk_in_order_and_filters_pdf(self):
        library = Mock()
        records = [('section ' + str(i) + ' x' * 2000,
                    {'source': 'full.pdf', 'page': i + 1}) for i in range(12)]
        library.records.return_value = records
        assistant = Assistant(library)
        assistant.generate = Mock(return_value='short cited notes')
        with patch('sys.stdout', new=io.StringIO()):
            assistant.ask('cities', 'Summarize this PDF', 'full.pdf')
        library.records.assert_called_with('cities', 'full.pdf')
        library.search.assert_not_called()
        prompts = '\n'.join(call.args[0] for call in assistant.generate.call_args_list)
        for i in range(12):
            self.assertIn(f'section {i} ', prompts)
        self.assertEqual([r for batch in batches(records) for r in batch], records)

    def test_chat_switches_topics_and_restores_pdf_selection(self):
        library = Mock()
        assistant = Mock()
        assistant.histories = {}
        assistant.ask.return_value = 'answer'
        commands = ['/pdf first.pdf', 'first question', '/use biology', 'second question',
                    '/use cities', 'followup', '/quit']
        with patch('builtins.input', side_effect=commands), patch('sys.stdout', new=io.StringIO()):
            chat(library, assistant, 'cities')
        self.assertEqual([call.args for call in assistant.ask.call_args_list], [
            ('cities', 'first question', 'first.pdf'),
            ('biology', 'second question', None),
            ('cities', 'followup', 'first.pdf')])

    def test_multi_pdf_comparison_happens_only_after_document_summaries(self):
        library = Mock()
        library.records.return_value = [
            ('first document evidence', {'source': 'first.pdf', 'page': 1}),
            ('second document evidence', {'source': 'second.pdf', 'page': 1}),
        ]
        assistant = Assistant(library)
        assistant.generate = Mock(side_effect=['first document summary',
                                               'second document summary', 'comparison'])
        with patch('sys.stdout', new=io.StringIO()):
            answer = assistant.ask('cities', 'Compare both PDFs', summary=True)
        prompts = [call.args[0] for call in assistant.generate.call_args_list]
        self.assertIn('first document evidence', prompts[0])
        self.assertNotIn('second document evidence', prompts[0])
        self.assertIn('second document evidence', prompts[1])
        self.assertNotIn('Compare both PDFs', prompts[0] + prompts[1])
        self.assertIn('first document summary', prompts[2])
        self.assertIn('second document summary', prompts[2])
        self.assertIn('Compare both PDFs', prompts[2])
        self.assertIn('first.pdf — PDF pages 1', answer)
        self.assertIn('second.pdf — PDF pages 1', answer)

    def test_ollama_connection_failure_is_actionable(self):
        import requests
        with patch('requests.post', side_effect=requests.ConnectionError('offline')):
            with self.assertRaisesRegex(RuntimeError, 'ollama list'):
                Assistant(Mock()).generate('question')


if __name__ == '__main__':
    unittest.main()
