#!/usr/bin/env python3

import json
import os
import re
import time
from typing import Optional, List

import rospy

from chef_robot_assistant.srv import GenerateRecipe
from chef_robot_assistant.srv import GenerateRecipeResponse
from chef_robot_assistant.srv import ValidateDishName
from chef_robot_assistant.srv import ValidateDishNameResponse

try:
    import requests
except ImportError as exc:
    requests = None
    REQUESTS_IMPORT_ERROR = str(exc)
else:
    REQUESTS_IMPORT_ERROR = ""

try:
    from dotenv import dotenv_values
except ImportError as exc:
    dotenv_values = None
    DOTENV_IMPORT_ERROR = str(exc)
else:
    DOTENV_IMPORT_ERROR = ""


NODE_NAME = "recipe_node"
GEMINI_API_URL_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)

RECIPE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "dish_name": {"type": "string"},
        "spoken_summary": {"type": "string"},
        "missing_ingredients": {
            "type": "array",
            "items": {"type": "string"},
        },
        "full_recipe": {"type": "string"},
    },
    "required": ["dish_name", "spoken_summary", "full_recipe"],
}


class RecipeNode(object):
    def __init__(self):
        rospy.init_node(NODE_NAME)

        self.recipe_model_name = self._param("recipe_model_name", "gemini-2.5-flash-lite")
        self.recipe_api_timeout_sec = float(
            self._param("recipe_api_timeout_sec", 20.0)
        )
        self.recipe_temperature = float(self._param("recipe_temperature", 0.4))
        self.recipe_max_output_tokens = int(
            self._param("recipe_max_output_tokens", 512)
        )
        self.recipe_retry_buffer_sec = float(
            self._param("recipe_retry_buffer_sec", 5.0)
        )
        self.repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.dotenv_path = os.path.join(self.repo_root, ".env")

        rospy.Service("generate_recipe", GenerateRecipe, self.handle_recipe)
        rospy.Service(
            "validate_dish_name", ValidateDishName, self.handle_validate_dish_name
        )
        rospy.loginfo(
            "%s ready with Gemini model %s",
            NODE_NAME,
            self.recipe_model_name,
        )

    def _param(self, key, default):
        return rospy.get_param("/chef_robot_assistant/{0}".format(key), default)

    def _dependency_error_message(self):
        errors = []
        if REQUESTS_IMPORT_ERROR:
            errors.append("requests import failed: {0}".format(REQUESTS_IMPORT_ERROR))
        if DOTENV_IMPORT_ERROR:
            errors.append(
                "python-dotenv import failed: {0}".format(DOTENV_IMPORT_ERROR)
            )
        return "; ".join(errors)

    def _normalize_ingredient(self, ingredient):
        cleaned = re.sub(r"\s+", " ", str(ingredient).strip().lower())
        return cleaned

    def _normalize_cuisine(self, cuisine):
        return re.sub(r"\s+", " ", str(cuisine).strip().lower())

    def _read_gemini_api_key(self):
        dependency_error = self._dependency_error_message()
        if dependency_error:
            return "", dependency_error

        if not os.path.isfile(self.dotenv_path):
            return "", "dotenv file not found at {0}".format(self.dotenv_path)

        try:
            dotenv_map = dotenv_values(self.dotenv_path)
        except Exception as exc:
            return "", "failed to read dotenv file {0}: {1}".format(
                self.dotenv_path,
                exc,
            )

        api_key = str(dotenv_map.get("GEMINI_API_KEY", "")).strip()
        if not api_key:
            return "", "GEMINI_API_KEY is missing from {0}".format(self.dotenv_path)

        return api_key, ""

    def _normalize_match_text(self, text):
        cleaned = re.sub(r"[^a-z0-9\s]", " ", str(text).lower())
        return re.sub(r"\s+", " ", cleaned).strip()

    def _find_missing_detected_ingredients(
        self,
        ingredients,
        full_recipe_text,
        missing_ingredients,
    ):
        normalized_recipe = self._normalize_match_text(full_recipe_text)
        normalized_missing = [
            self._normalize_match_text(item) for item in missing_ingredients
        ]
        missing = []
        for item in ingredients:
            if self._text_contains_ingredient(" ".join(normalized_missing), item):
                missing.append(item)
                continue
            if item and not self._text_contains_ingredient(normalized_recipe, item):
                missing.append(item)
        return missing

    def _text_contains_ingredient(self, text, ingredient):
        normalized_text = self._normalize_match_text(text)
        normalized_ingredient = self._normalize_match_text(ingredient)
        if not normalized_text or not normalized_ingredient:
            return False
        pattern = r"(^|\s){0}($|\s)".format(re.escape(normalized_ingredient))
        return re.search(pattern, normalized_text) is not None

    def _build_prompt(
        self,
        ingredients,
        cuisine,
        requested_dish="",
        health_preference="",
        allergies="",
        strict=False,
        missing_required=None,
        only_use_detected=False,
    ):
        """Construct the prompt text for the Gemini recipe generation API.

        Builds a structured prompt that includes detected ingredients, cuisine,
        dietary restrictions, allergies, and optional strictness instructions.
        The prompt asks the model to return a JSON object with specific keys.

        Parameters
        ----------
        ingredients : list of str
            Detected ingredients to include in the recipe.
        cuisine : str
            Desired cuisine type (e.g., "italian", "chinese").
        requested_dish : str, optional
            Specific dish name the user wants to prepare. Defaults to "".
        health_preference : str, optional
            Dietary preference (e.g., "vegetarian", "low-sodium").
            Ignored if empty or equal to "no". Defaults to "".
        allergies : str, optional
            Comma-separated list of ingredients to exclude.
            Ignored if empty or equal to "no". Defaults to "".
        strict : bool, optional
            If True, requires every detected ingredient to appear in the
            recipe and references any previously missing ingredients.
            Defaults to False.
        missing_required : list of str, optional
            Ingredients that were missing in a previous attempt; only used
            when ``strict`` is True. Defaults to None.
        only_use_detected : bool, optional
            If True, disallows any extra ingredients beyond those detected.
            Defaults to False.

        Returns
        -------
        str
            The formatted prompt ready to be sent to the Gemini API.
        """
        ingredient_text = ", ".join(ingredients)
        cuisine_text = cuisine or "mixed"
        restriction_lines = ""
        if only_use_detected:
            restriction_lines = (
                "CRITICAL INSTRUCTION: You MUST generate a recipe using ONLY the detected ingredients.\n"
                "Do NOT add any extra ingredients. Basic pantry items are NOT allowed unless they are in the detected list.\n"
                "The 'Missing Ingredients' section MUST strictly say 'none'.\n"
            )
        else:
            restriction_lines = "Basic pantry items are allowed when needed.\n"

        health_lines = ""
        if health_preference and health_preference != "no":
            health_lines = (
                "The user has specified a {0} requirement. Please adjust the recipe "
                "ingredients and preparation to strictly reflect this preference.\n"
            ).format(health_preference)

        allergy_lines = ""
        if allergies and allergies.lower() != "no":
            allergy_lines = (
                "CRITICAL FATAL WARNING: The user has the following allergies: '{0}'.\n"
                "You MUST NOT include these ingredients, nor any derivatives of them, anywhere in the recipe.\n"
            ).format(allergies)

        strict_lines = ""
        if strict:
            strict_lines = (
                "Every detected ingredient MUST appear in the Ingredients list and be used in the steps.\n"
                "Do not omit any detected ingredient.\n"
            )
            if missing_required:
                strict_lines += "The previous response missed: {0}\n".format(
                    ", ".join(missing_required)
                )

        dish_lines = ""
        if requested_dish:
            dish_lines = (
                "The user specifically wants to make the dish: '{0}'.\n"
                "You must tailor the recipe output specifically to create this dish while using the detected ingredients.\n"
            ).format(requested_dish)

        return (
            "You are a practical cooking assistant for a home robot.\n"
            "Generate one concise recipe in English.\n"
            "Use the detected ingredients as the main ingredients.\n"
            "{health_lines}"
            "{allergy_lines}"
            "{dish_lines}"
            "{strict_lines}"
            "{restriction_lines}"
            "List only the extra ingredients the user must buy in Missing Ingredients.\n"
            "If no extra ingredients are needed, write 'none'.\n"
            "Keep the spoken summary short enough to be read aloud naturally.\n"
            "Keep the recipe practical and concise.\n\n"
            "Detected ingredients: {ingredient_text}\n"
            "Requested cuisine: {cuisine_text}\n\n"
            "Return the recipe as a JSON object with exactly these keys:\n"
            "- \"dish_name\": short dish name\n"
            "- \"spoken_summary\": 1-2 sentence summary\n"
            "- \"missing_ingredients\": list of extra ingredients needed (empty list if none)\n"
            "- \"full_recipe\": short ingredient list and numbered cooking steps"
        ).format(
            health_lines=health_lines,
            allergy_lines=allergy_lines,
            dish_lines=dish_lines,
            restriction_lines=restriction_lines,
            strict_lines=strict_lines,
            ingredient_text=ingredient_text,
            cuisine_text=cuisine_text,
        )

    def _call_gemini_api(self, api_key, prompt, allowed_responses: Optional[List[str]]=None, response_schema: Optional[dict]=None):
        """Call the Gemini API with an optional constrained response schema.

        Parameters
        ----------
        api_key : str
            Gemini API key.
        prompt : str
            Prompt text to send.
        allowed_responses : list of str, optional
            When provided, the API response is constrained via
            ``generationConfig.responseSchema`` so that the model returns
            exactly one of the allowed strings.
        response_schema : dict, optional
            When provided, the API response is constrained to valid JSON
            matching the supplied OpenAPI schema object.  The returned
            ``response_text`` will be the parsed dict.

        Returns
        -------
        tuple
            (response_text, error_message).  ``response_text`` is empty
            when ``error_message`` is non-empty.  If ``response_schema``
            is provided, ``response_text`` is a dict.
        """
        dependency_error = self._dependency_error_message()
        if dependency_error:
            return "", dependency_error

        url = GEMINI_API_URL_TEMPLATE.format(model=self.recipe_model_name)
        generation_config = {
            "temperature": self.recipe_temperature,
            "maxOutputTokens": self.recipe_max_output_tokens,
        }
        if allowed_responses:
            generation_config["responseMimeType"] = "text/x.enum"
            generation_config["responseSchema"] = {
                "type": "STRING",
                "enum": allowed_responses,
            }
        elif response_schema:
            generation_config["responseMimeType"] = "application/json"
            generation_config["responseSchema"] = response_schema
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt,
                        }
                    ]
                }
            ],
            "generationConfig": generation_config,
        }
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        }

        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.recipe_api_timeout_sec,
            )
        except requests.exceptions.Timeout:
            return "", "Gemini API request timed out"
        except requests.exceptions.RequestException as exc:
            return "", "Gemini API request failed: {0}".format(exc)

        try:
            response_json = response.json()
        except ValueError:
            response_json = None

        if response.status_code >= 400:
            error_message = ""
            if isinstance(response_json, dict):
                error_message = (
                    response_json.get("error", {}).get("message", "") or ""
                ).strip()
            if not error_message:
                error_message = response.text.strip() or "unknown Gemini API error"
            return "", "Gemini API returned {0}: {1}".format(
                response.status_code,
                error_message,
            )

        if not isinstance(response_json, dict):
            return "", "Gemini API returned a non-JSON response"

        if response_json.get("promptFeedback", {}).get("blockReason"):
            return "", "Gemini API blocked the prompt"

        text_parts = []
        for candidate in response_json.get("candidates", []):
            content = candidate.get("content", {})
            for part in content.get("parts", []):
                text = part.get("text")
                if text:
                    text_parts.append(text)

        combined_text = "\n".join(text_parts).strip()
        if not combined_text:
            return "", "Gemini API returned no text"

        if allowed_responses:
            try:
                parsed = json.loads(combined_text)
                if isinstance(parsed, str):
                    combined_text = parsed
            except (ValueError, TypeError):
                pass
            return combined_text, ""

        if response_schema:
            try:
                parsed = json.loads(combined_text)
                if not isinstance(parsed, dict):
                    return "", "Gemini API returned a non-object JSON response"
                return parsed, ""
            except (ValueError, TypeError) as exc:
                return "", "Gemini API returned invalid JSON: {0}".format(exc)

        return combined_text, ""

    def _parse_missing_ingredients(self, raw_text):
        cleaned = raw_text.strip()
        if not cleaned:
            return []

        normalized = cleaned.lower()
        if normalized in ("none", "nil", "n/a", "nothing"):
            return []

        items = []
        for line in cleaned.splitlines():
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"^[-*•\d\.\)\s]+", "", line).strip()
            if not line:
                continue
            if "," in line:
                parts = [part.strip() for part in line.split(",")]
                items.extend(part for part in parts if part)
            else:
                items.append(line)

        if len(items) == 1 and items[0].lower() in ("none", "nil", "n/a", "nothing"):
            return []

        unique_items = []
        seen = set()
        for item in items:
            normalized_item = item.lower()
            if normalized_item not in seen:
                seen.add(normalized_item)
                unique_items.append(item)
        return unique_items

    def _parse_recipe_sections(self, raw_text):
        sections = {
            "dish_name": [],
            "spoken_summary": [],
            "missing_ingredients": [],
            "full_recipe": [],
        }
        label_map = {
            "dish name:": "dish_name",
            "spoken summary:": "spoken_summary",
            "missing ingredients:": "missing_ingredients",
            "full recipe:": "full_recipe",
        }

        current_section = None
        for raw_line in raw_text.splitlines():
            line = raw_line.rstrip()
            stripped = line.strip()
            lowered = stripped.lower()

            matched_section = None
            for label, section_name in label_map.items():
                if lowered.startswith(label):
                    matched_section = section_name
                    current_section = section_name
                    remainder = stripped[len(label):].strip()
                    if remainder:
                        sections[section_name].append(remainder)
                    break

            if matched_section:
                continue

            if current_section is not None:
                sections[current_section].append(stripped)

        dish_name = " ".join(line for line in sections["dish_name"] if line).strip()
        spoken_summary = " ".join(
            line for line in sections["spoken_summary"] if line
        ).strip()
        full_recipe_text = "\n".join(
            line for line in sections["full_recipe"] if line
        ).strip()
        missing_ingredients = self._parse_missing_ingredients(
            "\n".join(sections["missing_ingredients"])
        )

        if not dish_name:
            return None, "recipe output is missing 'Dish Name'"
        if not spoken_summary:
            return None, "recipe output is missing 'Spoken Summary'"
        if not full_recipe_text:
            return None, "recipe output is missing 'Full Recipe'"

        return {
            "dish_name": dish_name,
            "spoken_summary": spoken_summary,
            "full_recipe_text": full_recipe_text,
            "missing_ingredients": missing_ingredients,
        }, ""

    def handle_recipe(self, request):
        ingredients = []
        for item in request.ingredients:
            normalized = self._normalize_ingredient(item)
            if normalized:
                ingredients.append(normalized)
        cuisine = self._normalize_cuisine(request.cuisine)
        requested_dish = str(request.requested_dish).strip()
        health_preference = re.sub(r"\s+", " ", str(request.health_preference).strip().lower())
        allergies = str(request.allergies).strip()
        strict_ingredients = request.strict_ingredients

        if not ingredients:
            return GenerateRecipeResponse(
                success=False,
                dish_name="",
                spoken_summary="",
                full_recipe_text="",
                missing_ingredients=[],
                message="no ingredients were provided",
            )

        api_key, api_key_error = self._read_gemini_api_key()
        if not api_key:
            rospy.logwarn(api_key_error)
            return GenerateRecipeResponse(
                success=False,
                dish_name="",
                spoken_summary="",
                full_recipe_text="",
                missing_ingredients=[],
                message=api_key_error,
            )

        missing_required = []
        for attempt in range(2):
            if attempt > 0 and self.recipe_retry_buffer_sec > 0.0:
                rospy.loginfo(
                    "Waiting %.1f seconds before stricter Gemini recipe retry",
                    self.recipe_retry_buffer_sec,
                )
                time.sleep(self.recipe_retry_buffer_sec)
            prompt = self._build_prompt(
                ingredients,
                cuisine,
                requested_dish=requested_dish,
                health_preference=health_preference,
                allergies=allergies,
                strict=(attempt > 0),
                missing_required=missing_required,
                only_use_detected=strict_ingredients
            )
            recipe_data, recipe_error = self._call_gemini_api(
                api_key, prompt, response_schema=RECIPE_RESPONSE_SCHEMA
            )
            if recipe_error:
                rospy.logwarn(recipe_error)
                return GenerateRecipeResponse(
                    success=False,
                    dish_name="",
                    spoken_summary="",
                    full_recipe_text="",
                    missing_ingredients=[],
                    message=recipe_error,
                )

            if not isinstance(recipe_data, dict):
                rospy.logwarn("Unexpected Gemini recipe response format")
                return GenerateRecipeResponse(
                    success=False,
                    dish_name="",
                    spoken_summary="",
                    full_recipe_text="",
                    missing_ingredients=[],
                    message="unexpected API response format",
                )

            dish_name = str(recipe_data.get("dish_name", "")).strip()
            spoken_summary = str(recipe_data.get("spoken_summary", "")).strip()
            full_recipe_text = str(recipe_data.get("full_recipe", "")).strip()
            missing_ingredients_raw = recipe_data.get("missing_ingredients", [])
            if isinstance(missing_ingredients_raw, list):
                missing_ingredients_text = "\n".join(
                    str(item) for item in missing_ingredients_raw
                )
            else:
                missing_ingredients_text = str(missing_ingredients_raw)
            missing_ingredients = self._parse_missing_ingredients(
                missing_ingredients_text
            )

            if not dish_name:
                return GenerateRecipeResponse(
                    success=False,
                    dish_name="",
                    spoken_summary="",
                    full_recipe_text="",
                    missing_ingredients=[],
                    message="recipe output is missing 'dish_name'",
                )
            if not spoken_summary:
                return GenerateRecipeResponse(
                    success=False,
                    dish_name="",
                    spoken_summary="",
                    full_recipe_text="",
                    missing_ingredients=[],
                    message="recipe output is missing 'spoken_summary'",
                )
            if not full_recipe_text:
                return GenerateRecipeResponse(
                    success=False,
                    dish_name="",
                    spoken_summary="",
                    full_recipe_text="",
                    missing_ingredients=[],
                    message="recipe output is missing 'full_recipe'",
                )

            parsed_recipe = {
                "dish_name": dish_name,
                "spoken_summary": spoken_summary,
                "full_recipe_text": full_recipe_text,
                "missing_ingredients": missing_ingredients,
            }

            missing_required = self._find_missing_detected_ingredients(
                ingredients,
                parsed_recipe["full_recipe_text"],
                parsed_recipe["missing_ingredients"],
            )
            if not missing_required:
                rospy.loginfo(
                    "Generated recipe '%s' for cuisine=%s ingredients=%s dish=%s",
                    parsed_recipe["dish_name"],
                    cuisine or "mixed",
                    ", ".join(ingredients),
                    requested_dish or "none"
                )
                return GenerateRecipeResponse(
                    success=True,
                    dish_name=parsed_recipe["dish_name"],
                    spoken_summary=parsed_recipe["spoken_summary"],
                    full_recipe_text=parsed_recipe["full_recipe_text"],
                    missing_ingredients=parsed_recipe["missing_ingredients"],
                    message="recipe generation succeeded",
                )

            rospy.logwarn(
                "Recipe missing detected ingredients: %s",
                ", ".join(missing_required),
            )

        return GenerateRecipeResponse(
            success=False,
            dish_name="",
            spoken_summary="",
            full_recipe_text="",
            missing_ingredients=[],
            message=(
                "recipe output missing detected ingredients: {0}".format(
                    ", ".join(missing_required)
                )
            ),
        )

    def handle_validate_dish_name(self, request):
        """Validate whether a dish name is a real food dish using Gemini.

        Parameters
        ----------
        request : ValidateDishNameRequest
            Contains the ``dish_name`` string to validate.

        Returns
        -------
        ValidateDishNameResponse
            Fields:
            - ``is_valid`` (bool): True if Gemini confirms the dish is valid,
              False otherwise.
            - ``success`` (bool): True if the API call completed without error.
            - ``message`` (str): Status or error description.
        """
        dish_name = str(request.dish_name).strip()
        if not dish_name:
            return ValidateDishNameResponse(
                is_valid=False,
                success=True,
                message="empty dish name",
            )

        api_key, api_key_error = self._read_gemini_api_key()
        if not api_key:
            rospy.logwarn(api_key_error)
            return ValidateDishNameResponse(
                is_valid=True,
                success=False,
                message=api_key_error,
            )

        prompt = (
            'Is "{0}" the name of a real, edible food dish? '
            'Answer with only the word "yes" or "no".'.format(dish_name)
        )

        raw_text, error = self._call_gemini_api(
            api_key, prompt, allowed_responses=["yes", "no"]
        )
        if error:
            rospy.logwarn("Dish validation API error: %s", error)
            return ValidateDishNameResponse(
                is_valid=True,
                success=False,
                message=error,
            )

        normalized = raw_text.strip().lower()
        is_valid = normalized.startswith("yes")

        rospy.loginfo(
            "Dish validation for '%s': %s (raw='%s')",
            dish_name,
            is_valid,
            raw_text.strip(),
        )

        return ValidateDishNameResponse(
            is_valid=is_valid,
            success=True,
            message="validation succeeded",
        )


def main():
    RecipeNode()
    rospy.spin()


if __name__ == "__main__":
    main()
