# HTML 입력양식

#### 입력태그

-  server로 입력 받은 값을 전달하려면 반드시 form 태그 안에 input 태그를 입력해야 함.

``` <form action="입력한 데이터를 받아 처리할 스크립트 주소" method="post or get"/>
<form action="입력한 데이터를 받아 처리할 스크립트 주소" method="post or get"> <input ../> </form>
```

- input type 속성

``` &lt;input type = &quot;type 속성&quot; value=&quot;표시되는 텍스트&quot; name=&quot;서버에 전달되는 이름&quot;/&gt;
<input type = "type 속성" value="표시되는 텍스트"
	   name="서버에 전달되어 변수명처럼 사용"/>
```

![1559124152824](C:\Users\student\AppData\Roaming\Typora\typora-user-images\1559124152824.png)

- tel : pattern 속성에 정규식 지정, title에 올바른 입력 형식 지정

  ``` <input type ="tel" pattern"[0-9]{3}-[0-9]{4}-[0-9]{4}">
  <input type ="tel" pattern"[0-9]{3}-[0-9]{4}-[0-9]{4}">
  ```

- radio : name이 같아야 동일한 그룹으로 취급

  ​			반드시 name과 value를 지정해야 함, 중복 선택 X

- checkbox :  name이 같아야 동일한 그룹으로 취급

  ​					중복 선택 O, checked 옵션으로 미리 선택도 가능

- button type 

  - "submit" : 제출 버튼으로 입력된 데이터를 서버로 제출할 수 있음

  - input 태그로 작성된 이미지 버튼은 항상 제출 버튼의 역할만 함

    ``` <input type ="image" src = "xxx.jpg" alt = "제출버튼">
    <input type ="image" src = "xxx.jpg" alt = "제출버튼">
    ```

    -----------------------------------------------------------

    

- select : 항상 option 요소와 함께 사용하고 option은 반드시 value 값을 가져야 함.

``` <select multiple = "multiple">
<select multiple = "multiple">
<option value =""> ...</option>
<option value =""> ...</option> </select>
```

- 정규식 : http://regexlib.com 참고