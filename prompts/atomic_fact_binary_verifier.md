## **Task Description**

You will be given an **argument** and a **summary**. Your task is to evaluate whether the argument is supported by the summary and return a valid tuple in the specified format.

### **Decision Guidelines**

* **(1, "supported")** → The argument is **fully supported** by the summary.
* **(0, "missing")** → The argument **cannot be fully inferred** from the summary.
* **(0, "not-factual")** → The summary contains the information in a **contradictory** or **factually incorrect** way (e.g., misrepresented logical relationships, entity or terminology mismatches, or any other factual error).

## **Output Format**

First, briefly state your reasoning in plain text.
Then, return a valid **dictionary** with the key `"decision"` and the corresponding tuple as specified above. Do not include any additional text after the dictionary.

### **Example Output**

Brief reasoning in text.
{{"decision": (1, "supported")}}

## **Input**

* **Argument:**
  {argument}
* **Summary:**
  {summary}

State your reasoning first, then generate the dictionary.
