from process import CornellMovieCorpusProcessor


processor = CornellMovieCorpusProcessor(movie_lines_filepath,
                                        movie_conversations_filepath)
id2lines = processor.get_id2line()
conversations = processor.get_conversations()
questions, answers = processor.get_question_answer_set(id2lines, conversations)
result_filepaths = processor.prepare_seq2seq_files(questions, answers, args.output_directory)