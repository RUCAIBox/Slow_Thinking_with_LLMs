curl http://127.0.0.1:8021/v1/ \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-f89839f5a1dc426095708ef864c08875" \
  -d '{
        "model": "qwq",
        "messages": [
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": "Who was the king of England when Isaac Newton first published his Principia?"}
        ],
        "stream": false
      }'