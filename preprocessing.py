import os
import json
from multiprocessing import Pool

def process_json_file(args):
    input_filepath, output_directory = args
    try:
        with open(input_filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # "version" 필드 삭제
        if "version" in data:
            del data["version"]

        # "info" 객체에서 "위기단계"만 남기고 나머지 필드 삭제
        info = data.get('info', {})
        data['info'] = {"위기단계": info.get("위기단계", "")}

        # 아이의 대답과 상담사의 질문을 묶어서 새로운 JSON 형식으로 변환
        dialogues = []

        def collect_dialogues(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key == "audio" and isinstance(value, list):
                        for i in range(len(value)):
                            if value[i].get('type') == 'Q':
                                question = value[i].get('text', '')
                                answer = ''
                                # 다음 항목이 존재하고, 그것이 A 타입인지 확인
                                if i + 1 < len(value) and value[i + 1].get('type') == 'A':
                                    answer = value[i + 1].get('text', '')
                                dialogues.append({"Q": question, "A": answer})
                    elif isinstance(value, (dict, list)):
                        collect_dialogues(value)
            elif isinstance(obj, list):
                for item in obj:
                    collect_dialogues(item)

        collect_dialogues(data)

        # 합친 아이의 말, "kids_ment" 필드에 추가
        data["kids_ment"] = dialogues

        # "list" 필드를 삭제
        if "list" in data:
            del data["list"]

        # 출력 파일 경로 설정
        output_filepath = os.path.join(output_directory, os.path.basename(input_filepath))

        # 전처리 완료 파일 저장
        with open(output_filepath, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

        print(f"{input_filepath} successfully processed")
    except Exception as e:
        print(f"Failed to process {input_filepath}: {e}")

def process_directory(input_directory, output_directory):
    # 저장 디렉토리가 존재하지 않으면 생성
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 초기 데이터의 모든 JSON 파일 경로를 리스트로 생성
    files = [(os.path.join(input_directory, filename), output_directory) for filename in os.listdir(input_directory) if filename.endswith(".json")]

    # 멀티프로세싱을 사용하여 파일들을 병렬로 처리
    with Pool() as pool:
        pool.map(process_json_file, files)

if __name__ == '__main__':
    input_directory = 'C:/Users/LG/Downloads/data/Validation'
    output_directory = 'C:/Users/LG/Downloads/data/Val_data'
    
    process_directory(input_directory, output_directory)
