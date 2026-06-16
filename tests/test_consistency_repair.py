import pytest
from core.consistency_repair import repair_semantic_consistency

def test_fix_new_string_assignment():
    # Positive case
    legacy_code = """
    struct A {
        std::string name;
    };
    void init(A* a) {
        a->name = new std::string("test");
    }
    """
    fixed, repairs = repair_semantic_consistency(legacy_code)
    assert 'a->name = "test";' in fixed
    assert len(repairs) == 1

    # Negative case (already value assignment)
    correct_code = """
    struct A {
        std::string name;
    };
    void init(A* a) {
        a->name = "test";
    }
    """
    fixed_neg, repairs_neg = repair_semantic_consistency(correct_code)
    assert fixed_neg == correct_code
    assert len(repairs_neg) == 0

def test_fix_delete_on_string_members():
    # Positive case
    legacy_code = """
    struct A {
        std::string name;
    };
    void cleanup(A* a) {
        delete a->name;
    }
    """
    fixed, repairs = repair_semantic_consistency(legacy_code)
    assert "delete a->name;" not in fixed
    assert len(repairs) == 1

    # Negative case (delete on raw pointer)
    correct_code = """
    struct A {
        char* name;
    };
    void cleanup(A* a) {
        delete a->name;
    }
    """
    fixed_neg, repairs_neg = repair_semantic_consistency(correct_code)
    assert "delete a->name;" in fixed_neg
    assert len(repairs_neg) == 0

def test_fix_free_on_string_members():
    # Positive case
    legacy_code = """
    struct A {
        std::string name;
    };
    void cleanup(A* a) {
        free(a->name);
    }
    """
    fixed, repairs = repair_semantic_consistency(legacy_code)
    assert "free(a->name);" not in fixed
    assert len(repairs) == 1

    # Negative case (free on raw pointer)
    correct_code = """
    struct A {
        char* name;
    };
    void cleanup(A* a) {
        free(a->name);
    }
    """
    fixed_neg, repairs_neg = repair_semantic_consistency(correct_code)
    assert "free(a->name);" in fixed_neg
    assert len(repairs_neg) == 0

def test_fix_logger_fp_null():
    # Positive case
    legacy_code = """
    void log(const std::string& msg) {
        fprintf(fp, "%s\\n", msg);
    }
    """
    fixed, repairs = repair_semantic_consistency(legacy_code)
    assert 'fprintf(fp, "%s\\n", msg.c_str());' in fixed
    assert len(repairs) == 1

    # Negative case (already using .c_str())
    correct_code = """
    void log(const std::string& msg) {
        fprintf(fp, "%s\\n", msg.c_str());
    }
    """
    fixed_neg, repairs_neg = repair_semantic_consistency(correct_code)
    assert fixed_neg == correct_code
    assert len(repairs_neg) == 0

def test_fix_delete_on_unique_ptr_elements():
    # Positive case
    legacy_code = """
    struct A {
        std::vector<std::unique_ptr<int>> items;
    };
    void cleanup(A* a) {
        for(auto& i : a->items) {
            delete i;
        }
    }
    """
    fixed, repairs = repair_semantic_consistency(legacy_code)
    assert "delete i;" not in fixed
    assert len(repairs) == 1

    # Negative case (delete on raw pointers without unique_ptr)
    correct_code = """
    void cleanup() {
        int* p = new int;
        delete p;
    }
    """
    fixed_neg, repairs_neg = repair_semantic_consistency(correct_code)
    assert "delete p;" in fixed_neg
    assert len(repairs_neg) == 0

def test_fix_unique_ptr_return():
    # Positive case
    legacy_code = """
    int* find() {
        std::vector<std::unique_ptr<int>> items;
        for(const auto& i : items) {
            return i;
        }
    }
    """
    fixed, repairs = repair_semantic_consistency(legacy_code)
    assert "return i.get();" in fixed
    assert "return i;" not in fixed
    assert len(repairs) == 1

    # Negative case (correctly returning .get())
    correct_code = """
    int* find() {
        std::vector<std::unique_ptr<int>> items;
        for(const auto& i : items) {
            return i.get();
        }
    }
    """
    fixed_neg, repairs_neg = repair_semantic_consistency(correct_code)
    assert "return i.get();" in fixed_neg
    assert len(repairs_neg) == 0

def test_fix_cstr_in_c_functions():
    # Positive case
    legacy_code = """
    struct A {
        std::string name;
    };
    void print(A* a) {
        printf("%s", a->name);
        printf("%s", *a->name);
    }
    """
    fixed, repairs = repair_semantic_consistency(legacy_code)
    assert 'printf("%s", a->name.c_str());' in fixed
    assert '*a->name' not in fixed
    assert len(repairs) == 2

    # Negative case (already using .c_str() and no deref)
    correct_code = """
    struct A {
        std::string name;
    };
    void print(A* a) {
        printf("%s", a->name.c_str());
    }
    """
    fixed_neg, repairs_neg = repair_semantic_consistency(correct_code)
    assert fixed_neg == correct_code
    assert len(repairs_neg) == 0

def test_fix_callback_type_mismatch():
    # Positive case
    legacy_code = """
    typedef void (*Cb)(const char*);
    struct A {
        std::string name;
        Cb callback;
    };
    void notify(A* a) {
        a->callback(a->name);
    }
    """
    fixed, repairs = repair_semantic_consistency(legacy_code)
    assert "a->callback(a->name.c_str());" in fixed
    assert len(repairs) == 1

    # Negative case
    correct_code = """
    typedef void (*Cb)(const char*);
    struct A {
        std::string name;
        Cb callback;
    };
    void notify(A* a) {
        a->callback(a->name.c_str());
    }
    """
    fixed_neg, repairs_neg = repair_semantic_consistency(correct_code)
    assert fixed_neg == correct_code
    assert len(repairs_neg) == 0

def test_multiple_repairs():
    # Integration test testing multiple categories at once
    legacy_code = """
    struct A {
        std::string name;
        std::vector<std::unique_ptr<int>> items;
    };
    void cleanup(A* a) {
        delete a->name;
        for (auto& i : a->items) {
            delete i;
        }
        printf("%s", a->name);
    }
    """
    fixed, repairs = repair_semantic_consistency(legacy_code)
    assert "delete a->name;" not in fixed
    assert "delete i;" not in fixed
    assert 'printf("%s", a->name.c_str());' in fixed
    assert len(repairs) == 3
