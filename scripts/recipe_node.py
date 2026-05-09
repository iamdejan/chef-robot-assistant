#!/usr/bin/env python3

import rospy

from chef_robot_assistant.srv import GenerateRecipe
from chef_robot_assistant.srv import GenerateRecipeResponse


NODE_NAME = "recipe_node"


class RecipeNode(object):
    def __init__(self):
        rospy.init_node(NODE_NAME)
        rospy.Service("generate_recipe", GenerateRecipe, self.handle_recipe)
        rospy.loginfo("%s stub service is ready", NODE_NAME)

    def handle_recipe(self, request):
        ingredients = list(request.ingredients)
        cuisine = request.cuisine.strip().lower()

        if not ingredients:
            return GenerateRecipeResponse(
                success=False,
                dish_name="",
                spoken_summary="",
                full_recipe_text="",
                missing_ingredients=[],
                message="No ingredients were provided"
            )

        missing_ingredients = ["oil", "salt"]
        dish_name = "{0} ingredient stir fry".format(cuisine or "mixed").title()
        spoken_summary = "I created a simple {0} recipe using {1}.".format(
            cuisine or "mixed cuisine",
            ", ".join(ingredients)
        )
        full_recipe_text = (
            "Dish: {0}\n"
            "Cuisine: {1}\n"
            "Detected ingredients: {2}\n"
            "Missing pantry ingredients to buy: {3}\n"
            "Steps:\n"
            "1. Prepare the detected ingredients.\n"
            "2. Add the missing pantry ingredients.\n"
            "3. Cook everything together and serve."
        ).format(
            dish_name,
            cuisine or "mixed",
            ", ".join(ingredients),
            ", ".join(missing_ingredients)
        )

        return GenerateRecipeResponse(
            success=True,
            dish_name=dish_name,
            spoken_summary=spoken_summary,
            full_recipe_text=full_recipe_text,
            missing_ingredients=missing_ingredients,
            message="Foundation stub recipe response"
        )


def main():
    RecipeNode()
    rospy.spin()


if __name__ == "__main__":
    main()
